import json

import pytest

from evals.dataset import load_questions
from evals.fake_judge import FakeJudge
from evals.judge import JudgeParseError, build_judge_message, format_sources_for_judge, parse_judge_response
from evals.metrics import QuestionResult, Summary, answer_contains_phrase, score_one, summarize
from evals.report import write_report
from evals.run import check_regression, main, run_eval
from rag.fake_llm import FakeAnthropic
from rag.generation import ABSTAIN_TEXT, AnswerGenerator, make_generator
from rag.pipeline import RAGPipeline
from tests.helpers import ScriptedClient, sample_index


def fake_pipeline() -> RAGPipeline:
    return RAGPipeline(sample_index(), make_generator("fake"))


def make_answer(text: str, sources=None, warnings=None, answered=True):
    from rag.pipeline import Answer

    return Answer(
        question="q", answer=text, answered=answered, citations=[1] if answered else [],
        sources=sources or [{"n": 1, "source": "a.md", "section": "A", "text": "Tents last 5 years."}],
        warnings=warnings or [],
    )


# ---------------------------------------------------------------- dataset validation

def test_bundled_eval_questions_are_valid():
    questions = load_questions()
    assert len(questions) >= 20
    assert {q["tag"] for q in questions} >= {"keyword", "paraphrase", "unanswerable"}


def test_load_questions_rejects_a_question_missing_a_tag(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps({"question": "Hi?", "answer_contains": "x"}) + "\n")
    with pytest.raises(ValueError, match="tag"):
        load_questions(path)


def test_load_questions_rejects_an_answerable_question_missing_answer_contains(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps({"question": "Hi?", "tag": "keyword"}) + "\n")
    with pytest.raises(ValueError, match="answer_contains"):
        load_questions(path)


def test_load_questions_allows_unanswerable_questions_without_answer_contains(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps({"question": "Hi?", "tag": "unanswerable", "answerable": False}) + "\n")
    assert load_questions(path)[0]["question"] == "Hi?"


# ---------------------------------------------------------------- judge prompt and parsing

def test_judge_message_includes_sources_question_and_answer():
    message = build_judge_message("How long?", "[1] (source: a.md | section: A)\nFive years.", "Five years [1].")
    assert "[1] (source: a.md" in message and "Question: How long?" in message and "Five years [1]." in message


def test_parse_judge_response_handles_clean_and_noisy_json():
    result = parse_judge_response('{"faithful": true, "relevant": false, "reason": "partial"}')
    assert result.faithful is True and result.relevant is False and result.reason == "partial"

    noisy = parse_judge_response('Sure, here it is:\n{"faithful": false, "relevant": true}\nHope that helps!')
    assert noisy.faithful is False and noisy.relevant is True


def test_parse_judge_response_rejects_missing_fields_or_bad_json():
    with pytest.raises(JudgeParseError, match="No JSON"):
        parse_judge_response("I think it's faithful.")
    with pytest.raises(JudgeParseError, match="not valid JSON"):
        parse_judge_response("{faithful: true}")
    with pytest.raises(JudgeParseError, match="missing required fields"):
        parse_judge_response('{"faithful": true}')


def test_format_sources_for_judge_numbers_every_source():
    answer = make_answer("x", sources=[
        {"n": 1, "source": "a.md", "section": "A", "text": "one"},
        {"n": 2, "source": "b.md", "section": "B", "text": "two"},
    ])
    block = format_sources_for_judge(answer)
    assert "[1] (source: a.md | section: A)\none" in block
    assert "[2] (source: b.md | section: B)\ntwo" in block


# ---------------------------------------------------------------- LLMJudge (scripted client, no network)

def test_llm_judge_sends_the_right_prompt_and_parses_the_reply():
    from evals.judge import LLMJudge

    client = ScriptedClient('{"faithful": true, "relevant": true, "reason": "matches source 1"}')
    result = LLMJudge(client=client, model="judge-model").judge("How long?", "[1] ...", "Five years [1].")
    assert result.faithful and result.relevant and "source 1" in result.reason
    call = client.calls[0]
    assert call["model"] == "judge-model" and call["temperature"] == 0
    assert "Five years [1]." in call["messages"][0]["content"]


def test_llm_judge_wraps_provider_errors():
    from evals.judge import LLMJudge
    from rag.generation import LLMError
    from tests.helpers import FailingClient

    with pytest.raises(LLMError, match="judge request failed"):
        LLMJudge(client=FailingClient()).judge("q", "[1] ...", "answer")


# ---------------------------------------------------------------- fake judge

def test_fake_judge_marks_abstentions_faithful_and_relevant():
    result = FakeJudge().judge("Do you sell kayaks?", "[1] ...", ABSTAIN_TEXT)
    assert result.faithful and result.relevant


def test_fake_judge_flags_an_answer_that_shares_no_words_with_its_sources():
    result = FakeJudge().judge("What is the password minimum?", "Tents last five years.", "The sky is blue today.")
    assert not result.faithful


def test_fake_judge_marks_a_grounded_answer_faithful():
    result = FakeJudge().judge(
        "How long do tents last under warranty?",
        "[1] Tents and backpacks: 5 years.",
        "Tents are covered for 5 years. [1]",
    )
    assert result.faithful and result.relevant


# ---------------------------------------------------------------- metrics

def test_answer_contains_phrase_is_case_and_whitespace_insensitive():
    assert answer_contains_phrase("Tents last   FIVE years total.", "five years")
    assert not answer_contains_phrase("Tents last three years.", "five years")


def test_score_one_flags_wrong_abstention_decision():
    record = {"question": "q", "tag": "keyword", "answer_contains": "5 years"}
    result = score_one(record, make_answer(ABSTAIN_TEXT, answered=False), None)
    assert not result.correct_abstention and result.correct_content is None and not result.passed


def test_score_one_checks_content_only_when_answered_and_expected():
    record = {"question": "q", "tag": "keyword", "answer_contains": "5 years"}
    right = score_one(record, make_answer("Tents last 5 years. [1]"), None)
    wrong = score_one(record, make_answer("Tents last 3 years. [1]"), None)
    assert right.correct_content and right.passed
    assert wrong.correct_content is False and not wrong.passed


def test_score_one_skips_content_check_for_unanswerable_questions():
    record = {"question": "q", "tag": "unanswerable", "answerable": False}
    result = score_one(record, make_answer(ABSTAIN_TEXT, answered=False), None)
    assert result.correct_content is None and result.correct_abstention and result.passed


def test_score_one_fails_on_invalid_citation_warning():
    record = {"question": "q", "tag": "keyword", "answer_contains": "5 years"}
    answer = make_answer("Tents last 5 years. [1][9]", warnings=["The answer cites source numbers that do not exist: [9]"])
    result = score_one(record, answer, None)
    assert not result.citation_valid and not result.passed


def test_score_one_requires_the_judge_to_agree_when_a_judge_is_used():
    record = {"question": "q", "tag": "keyword", "answer_contains": "5 years"}
    from evals.judge import JudgeResult

    unfaithful = JudgeResult(faithful=False, relevant=True, reason="invented a detail", raw="")
    result = score_one(record, make_answer("Tents last 5 years. [1]"), unfaithful)
    assert result.correct_content and not result.passed  # content matched, but the judge still fails it


def test_summarize_computes_rates_and_breaks_down_by_tag():
    records = [
        {"question": "a", "tag": "keyword", "answer_contains": "5 years"},
        {"question": "b", "tag": "keyword", "answer_contains": "5 years"},
        {"question": "c", "tag": "unanswerable", "answerable": False},
    ]
    answers = [make_answer("Tents last 5 years. [1]"), make_answer("Tents last 3 years. [1]"), make_answer(ABSTAIN_TEXT, answered=False)]
    results = [score_one(r, a, None) for r, a in zip(records, answers)]
    summary = summarize(results)
    assert summary.n == 3
    assert summary.content_accuracy == pytest.approx(0.5)
    assert summary.by_tag["keyword"]["content_accuracy"] == pytest.approx(0.5)
    assert summary.by_tag["unanswerable"]["content_accuracy"] is None
    assert summary.judge_faithful_rate is None  # no judge was used


def test_summary_to_dict_round_trips_through_json():
    summary = Summary(n=1, pass_rate=1.0, abstention_accuracy=1.0, content_accuracy=None,
                       citation_validity_rate=1.0, judge_faithful_rate=None, judge_relevant_rate=None, by_tag={})
    assert json.loads(json.dumps(summary.to_dict()))["pass_rate"] == 1.0


# ---------------------------------------------------------------- report

def test_write_report_lists_failures_and_the_metrics_table(tmp_path):
    record = {"question": "How long is the tent warranty?", "tag": "keyword", "answer_contains": "5 years"}
    result = score_one(record, make_answer("Tents last 3 years. [1]"), None)
    write_report([result], summarize([result]), tmp_path / "report.md")
    text = (tmp_path / "report.md").read_text()
    assert "How long is the tent warranty?" in text and "Overall pass rate" in text


def test_write_report_says_so_when_nothing_failed(tmp_path):
    record = {"question": "q", "tag": "keyword", "answer_contains": "5 years"}
    result = score_one(record, make_answer("Tents last 5 years. [1]"), None)
    write_report([result], summarize([result]), tmp_path / "report.md")
    assert "Every question passed" in (tmp_path / "report.md").read_text()


# ---------------------------------------------------------------- run_eval end to end (fake pipeline + fake judge)

def test_run_eval_end_to_end_with_fake_components():
    questions = load_questions()[:6]
    outcome = run_eval(fake_pipeline(), FakeJudge(), questions, mode="hybrid", top_k=4, verbose=False)
    assert len(outcome["results"]) == 6
    assert outcome["summary"].n == 6
    assert outcome["summary"].judge_faithful_rate is not None  # a judge was supplied


def test_run_eval_works_without_a_judge():
    questions = load_questions()[:3]
    outcome = run_eval(fake_pipeline(), None, questions, mode="hybrid", top_k=4, verbose=False)
    assert all(r.judge is None for r in outcome["results"])
    assert outcome["summary"].judge_faithful_rate is None


# ---------------------------------------------------------------- regression check

def test_check_regression_passes_when_scores_match_or_improve():
    baseline = {"pass_rate": 0.80, "citation_validity_rate": 1.0}
    assert check_regression({"pass_rate": 0.80, "citation_validity_rate": 1.0}, baseline, tolerance=0.05) == []
    assert check_regression({"pass_rate": 0.90, "citation_validity_rate": 1.0}, baseline, tolerance=0.05) == []


def test_check_regression_tolerates_small_drops_but_flags_big_ones():
    baseline = {"pass_rate": 0.80}
    assert check_regression({"pass_rate": 0.77}, baseline, tolerance=0.05) == []
    problems = check_regression({"pass_rate": 0.60}, baseline, tolerance=0.05)
    assert problems and "pass_rate dropped from 80.0% to 60.0%" in problems[0]


def test_check_regression_ignores_metrics_missing_from_either_side():
    assert check_regression({"pass_rate": 0.5}, {"judge_faithful_rate": None}, tolerance=0.05) == []
    assert check_regression({}, {"pass_rate": 0.9}, tolerance=0.05) == []


# ---------------------------------------------------------------- CLI

def test_cli_runs_writes_reports_and_can_save_a_baseline(tmp_path):
    index_dir = tmp_path / "index"
    sample_index().save(index_dir)
    reports_dir = tmp_path / "reports"
    baseline_path = tmp_path / "baseline.json"
    code = main([
        "--index", str(index_dir), "--llm", "fake", "--judge", "fake",
        "--reports-dir", str(reports_dir), "--save-as", str(baseline_path),
    ])
    assert code == 0
    assert (reports_dir / "eval_report.md").exists() and (reports_dir / "eval_results.json").exists()
    assert json.loads(baseline_path.read_text())["n"] == len(load_questions())


def test_cli_check_regression_fails_the_build_on_a_real_regression(tmp_path):
    index_dir = tmp_path / "index"
    sample_index().save(index_dir)
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"pass_rate": 0.999}))  # impossible to meet: forces a failure
    code = main([
        "--index", str(index_dir), "--llm", "fake", "--judge", "fake",
        "--reports-dir", str(tmp_path / "reports"), "--baseline", str(baseline_path), "--check-regression",
    ])
    assert code == 1


def test_cli_check_regression_passes_against_a_lenient_baseline(tmp_path):
    index_dir = tmp_path / "index"
    sample_index().save(index_dir)
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"pass_rate": 0.0}))
    code = main([
        "--index", str(index_dir), "--llm", "fake", "--judge", "fake",
        "--reports-dir", str(tmp_path / "reports"), "--baseline", str(baseline_path), "--check-regression",
    ])
    assert code == 0


def test_cli_check_regression_without_a_baseline_file_fails_clearly(tmp_path, capsys):
    index_dir = tmp_path / "index"
    sample_index().save(index_dir)
    code = main([
        "--index", str(index_dir), "--llm", "fake", "--judge", "fake",
        "--reports-dir", str(tmp_path / "reports"), "--baseline", str(tmp_path / "nope.json"), "--check-regression",
    ])
    assert code == 1
    assert "no baseline" in capsys.readouterr().err


def test_cli_reports_a_missing_index(tmp_path, capsys):
    code = main(["--index", str(tmp_path / "missing"), "--llm", "fake", "--judge", "fake"])
    assert code == 1
    assert "rag.ingest" in capsys.readouterr().err
