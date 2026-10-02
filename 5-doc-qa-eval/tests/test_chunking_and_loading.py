import pytest

from rag.chunking import chunk_document, split_sections, word_count
from rag.loaders import load_documents
from tests.helpers import make_doc


def test_section_path_follows_the_heading_hierarchy():
    doc = make_doc("guide.md", "# Guide\n\nIntro text.\n\n## Setup\n\nInstall it.\n\n### Linux\n\nUse apt.\n\n## Usage\n\nRun it.")
    sections = dict((path, body) for path, body in split_sections(doc))
    assert list(sections) == ["Guide", "Guide > Setup", "Guide > Setup > Linux", "Guide > Usage"]
    assert sections["Guide > Usage"] == "Run it."


def test_text_before_any_heading_uses_the_file_name_as_title():
    chunks = chunk_document(make_doc("notes/return-policy.txt", "Just plain text with no headings at all."))
    assert len(chunks) == 1
    assert chunks[0].section == "Return Policy"


def test_hash_lines_inside_code_fences_are_not_headings():
    doc = make_doc("a.md", "# Title\n\nRun:\n\n```\n# not a heading\nls\n```\n\nDone.")
    assert [path for path, _ in split_sections(doc)] == ["Title"]


def test_chunks_never_exceed_the_word_limit_and_lose_no_sentences():
    sentences = [f"Sentence number {i} talks about topic {i} in some detail." for i in range(60)]
    doc = make_doc("long.md", "# Long\n\n" + " ".join(sentences))
    chunks = chunk_document(doc, max_words=40, overlap_words=10)
    assert len(chunks) > 5
    assert all(word_count(c.text) <= 40 for c in chunks)
    joined = " ".join(c.text for c in chunks)
    assert all(s in joined for s in sentences)


def test_overlap_repeats_the_end_of_the_previous_chunk():
    sentences = [f"Fact {i} is stated here clearly." for i in range(30)]
    chunks = chunk_document(make_doc("o.md", "# T\n\n" + " ".join(sentences)), max_words=30, overlap_words=12)
    shared = 0
    for a, b in zip(chunks, chunks[1:]):
        last_sentence_of_a = a.text.split(". ")[-1]
        shared += last_sentence_of_a.rstrip(".") in b.text
    assert shared >= len(chunks) - 2  # nearly every boundary carries some overlap


def test_zero_overlap_means_chunks_share_no_sentences():
    sentences = [f"Fact {i} is stated here clearly." for i in range(30)]
    chunks = chunk_document(make_doc("o.md", "# T\n\n" + " ".join(sentences)), max_words=30, overlap_words=0)
    seen = set()
    for c in chunks:
        parts = set(c.text.split(". "))
        assert not (parts & seen)
        seen |= parts


def test_a_single_enormous_sentence_is_cut_by_words():
    doc = make_doc("x.md", "# T\n\n" + " ".join(f"w{i}" for i in range(250)))
    chunks = chunk_document(doc, max_words=100, overlap_words=0)
    assert [word_count(c.text) for c in chunks] == [100, 100, 50]


def test_long_bullet_lists_split_on_lines_and_keep_newlines():
    bullets = "\n".join(f"- item {i} has a few words in it" for i in range(40))
    chunks = chunk_document(make_doc("b.md", "# T\n\n" + bullets), max_words=50, overlap_words=0)
    assert len(chunks) > 3
    assert all(c.text.startswith("- item") for c in chunks)
    assert "\n- item" in chunks[0].text


def test_chunk_ids_are_unique_stable_and_ordered():
    doc = make_doc("doc.md", "# A\n\nOne.\n\n## B\n\nTwo.\n\n## C\n\nThree.")
    ids = [c.id for c in chunk_document(doc)]
    assert ids == ["doc.md#0", "doc.md#1", "doc.md#2"]
    assert ids == [c.id for c in chunk_document(doc)]


def test_contextual_text_includes_the_heading_path():
    chunk = chunk_document(make_doc("d.md", "# Policy\n\n## Refunds\n\nWithin 30 days."))[0]
    assert chunk.contextual_text.startswith("Policy > Refunds\n")


def test_invalid_chunk_settings_are_rejected():
    doc = make_doc("d.md", "text")
    with pytest.raises(ValueError, match="overlap_words"):
        chunk_document(doc, max_words=10, overlap_words=10)
    with pytest.raises(ValueError, match="max_words"):
        chunk_document(doc, max_words=0, overlap_words=0)


# ---------------------------------------------------------------- loading

def test_loader_reads_supported_files_recursively_and_skips_the_rest(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\nalpha")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("bravo")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "c.md").write_text("secret")
    docs = load_documents(tmp_path)
    assert [d.source for d in docs] == ["a.md", "sub/b.txt"]


def test_loader_errors_are_clear(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "nope")
    with pytest.raises(ValueError, match="No readable documents"):
        load_documents(tmp_path)
    (tmp_path / "blank.md").write_text("   \n\n")
    with pytest.raises(ValueError, match="No readable documents"):
        load_documents(tmp_path)  # blank files are skipped, leaving nothing
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(ValueError, match="UTF-8"):
        load_documents(tmp_path)


def test_loader_reads_pdfs(tmp_path):
    pytest.importorskip("pypdf")
    reportlab = pytest.importorskip("reportlab")
    from reportlab.pdfgen import canvas

    path = tmp_path / "policy.pdf"
    c = canvas.Canvas(str(path))
    c.drawString(72, 720, "Refunds are issued within 5 business days.")
    c.save()
    docs = load_documents(path)
    assert docs[0].source == "policy.pdf"
    assert "Refunds are issued" in docs[0].text
