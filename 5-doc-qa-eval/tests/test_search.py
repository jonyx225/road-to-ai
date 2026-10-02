import numpy as np
import pytest

from rag.bm25 import BM25, tokenize
from rag.embeddings import create_embedder
from rag.retrieval import Retriever, reciprocal_rank_fusion
from rag.store import Index, IndexNotFoundError, top_k_indices
from tests.helpers import make_doc, sample_index


# ---------------------------------------------------------------- BM25

def test_tokenizer_keeps_hyphenated_identifiers_whole_and_in_parts():
    tokens = tokenize("Error E-4102 on the NX-200!")
    assert "e-4102" in tokens and "4102" in tokens and "nx-200" in tokens and "200" in tokens
    assert "the" not in tokens and "on" not in tokens
    assert "e" not in tokens  # single letters are noise


def test_bm25_ranks_the_chunk_with_the_exact_identifier_first():
    index = sample_index()
    top = index.bm25.search("E-4102", 3)
    assert top and index.chunks[top[0][0]].text.startswith("- Error E-4102")


def test_bm25_returns_nothing_for_unknown_or_stopword_only_queries():
    bm25 = BM25(["the quick brown fox", "lazy dogs sleep"])
    assert bm25.search("kayaks", 5) == []
    assert bm25.search("the and of", 5) == []


def test_rare_words_outweigh_common_words():
    docs = ["apple apple apple banana", "banana cherry", "banana banana", "banana date"]
    bm25 = BM25(docs)
    best_index, _ = bm25.search("apple banana", 1)[0]
    assert best_index == 0
    assert bm25.idf["apple"] > bm25.idf["banana"]


def test_bm25_length_normalisation_prefers_the_shorter_document():
    bm25 = BM25(["refund policy", "refund policy " + "filler " * 200])
    ranked = bm25.search("refund", 2)
    assert [i for i, _ in ranked] == [0, 1]


# ---------------------------------------------------------------- embeddings and the index

def test_tfidf_vectors_are_unit_length():
    index = sample_index()
    norms = np.linalg.norm(index.embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_top_k_indices_orders_by_score_and_breaks_ties_by_position():
    scores = np.array([0.1, 0.9, 0.9, 0.5])
    assert top_k_indices(scores, 3).tolist() == [1, 2, 3]


def test_dense_search_finds_a_chunk_that_shares_its_wording():
    index = sample_index()
    i, _ = index.dense_search("How much does express shipping cost?", 1)[0]
    assert "Express shipping" in index.chunks[i].text


def test_index_round_trip_keeps_chunks_vectors_and_search_results(tmp_path):
    index = sample_index()
    index.save(tmp_path)
    loaded = Index.load(tmp_path)
    assert [c.id for c in loaded.chunks] == [c.id for c in index.chunks]
    assert np.array_equal(loaded.embeddings, index.embeddings)
    q = "warranty on backpacks"
    assert loaded.dense_search(q, 3) == index.dense_search(q, 3)
    assert loaded.bm25.search(q, 3) == index.bm25.search(q, 3)
    assert loaded.meta["documents"] == index.meta["documents"]


def test_loading_a_missing_index_explains_how_to_build_one(tmp_path):
    with pytest.raises(IndexNotFoundError, match="rag.ingest"):
        Index.load(tmp_path / "nothing-here")


def test_querying_with_the_wrong_embedder_is_caught():
    index = sample_index()

    class WrongDim:
        def encode_query(self, text):
            return np.ones(5, dtype=np.float32)

    swapped = Index(index.chunks, index.embeddings, WrongDim(), index.meta)
    with pytest.raises(ValueError, match="different embedder"):
        swapped.dense_search("anything", 3)


def test_building_an_index_from_empty_content_fails_clearly():
    with pytest.raises(ValueError, match="no chunks"):
        Index.build([make_doc("empty.md", "# Only a heading")], create_embedder("tfidf"))


def test_unknown_embedder_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown embedder"):
        create_embedder("word2vec")


def test_document_hashes_change_when_content_changes():
    a = Index.build([make_doc("d.md", "# T\n\nfirst version")], create_embedder("tfidf"))
    b = Index.build([make_doc("d.md", "# T\n\nsecond version")], create_embedder("tfidf"))
    assert a.meta["documents"]["d.md"]["sha256"] != b.meta["documents"]["d.md"]["sha256"]


# ---------------------------------------------------------------- retrieval

def test_reciprocal_rank_fusion_math():
    fused = reciprocal_rank_fusion([[10, 20, 30], [20, 10]], k=60)
    assert fused[20] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[10] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[30] == pytest.approx(1 / 63)


def test_a_chunk_ranked_well_by_both_retrievers_beats_one_ranked_first_by_only_one():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 3, 1]], k=60)
    assert max(fused, key=fused.get) == 2


def test_hybrid_reports_ranks_from_both_retrievers_for_an_exact_code():
    hits = Retriever(sample_index()).retrieve("What does error E-4102 mean?", mode="hybrid", top_k=3)
    top = hits[0]
    assert top.chunk.text.startswith("- Error E-4102")
    assert top.dense_rank is not None and top.bm25_rank == 1


def test_each_mode_fills_only_its_own_fields():
    r = Retriever(sample_index())
    dense = r.retrieve("tent warranty", mode="dense", top_k=3)
    lexical = r.retrieve("tent warranty", mode="bm25", top_k=3)
    assert all(h.bm25_rank is None and h.dense_rank is not None for h in dense)
    assert all(h.dense_rank is None and h.bm25_rank is not None for h in lexical)


def test_top_k_is_respected_and_modes_are_validated():
    r = Retriever(sample_index())
    assert len(r.retrieve("shipping", mode="hybrid", top_k=2)) == 2
    with pytest.raises(ValueError, match="mode"):
        r.retrieve("x", mode="semantic")
    with pytest.raises(ValueError, match="top_k"):
        r.retrieve("x", top_k=0)


def test_reranker_can_reorder_the_candidates():
    class PrefersRefunds:
        def score(self, query, passages):
            return [1.0 if "refund" in p.lower() else 0.0 for p in passages]

    r = Retriever(sample_index(), reranker=PrefersRefunds())
    hits = r.retrieve("shipping costs", mode="hybrid", top_k=3, rerank=True)
    assert all(h.rerank_score is not None for h in hits)
    assert "refund" in hits[0].chunk.text.lower()


def test_rerank_without_the_library_gives_a_clear_error():
    from rag.rerank import RerankerUnavailable

    try:
        import sentence_transformers  # noqa: F401
        pytest.skip("sentence-transformers is installed")
    except ImportError:
        pass
    with pytest.raises(RerankerUnavailable, match="sentence-transformers"):
        Retriever(sample_index()).retrieve("tent", rerank=True)
