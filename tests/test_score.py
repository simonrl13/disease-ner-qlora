from ner.data import bio_spans, gold_strings
from ner.score import counts, map_strings, parse_output, prf


def test_parse_output():
    assert parse_output(' ["a b", "c"] \n', False) == ["a b", "c"]
    assert parse_output("[]", False) == []
    assert parse_output('```json\n["a"]\n```', False) == ["a"]
    assert parse_output('```\n["a"]\n```', False) == ["a"]
    assert parse_output('```json\n```json\n["a"]\n```\n```', False) is None  # only one fence stripped
    assert parse_output('["a"]', True) is None  # stopped at max_new_tokens
    for bad in ['Diseases: ["a"]', '{"a": 1}', '["a", 1]', '"a"', "", '["a"] trailing']:
        assert parse_output(bad, False) is None, bad

TOK = "Breast cancer and cancer , not cancers or Cancer .".split()


def test_bio_spans_stray_inside_starts_span():
    assert bio_spans(["I-Disease", "I-Disease", "O", "B-Disease", "B-Disease", "I-Disease"]) == [(0, 2), (3, 4), (4, 6)]
    assert bio_spans(["O", "O"]) == []


def test_whole_token_case_sensitive():
    # "cancer" matches tokens 1 and 3 only: not "cancers" (partial token) nor "Cancer" (case)
    assert map_strings(TOK, ["cancer"]) == ([(1, 2), (3, 4)], 0)
    assert map_strings(TOK, ["ancer"]) == ([], 1)
    assert map_strings(TOK, ["CANCER"]) == ([], 1)


def test_longer_first_and_no_reuse():
    spans, ph = map_strings(TOK, ["cancer", "Breast cancer"])
    assert spans == [(0, 2), (3, 4)] and ph == 0


def test_duplicates_empty_and_phantoms():
    assert map_strings(TOK, ["cancer", "cancer"]) == ([(1, 2), (3, 4)], 0)
    assert map_strings(TOK, ["", "  "]) == ([], 2)
    # "Breast cancer" claims tokens 0-1; "Breast" then has no free occurrence -> phantom
    assert map_strings(TOK, ["Breast cancer", "Breast"]) == ([(0, 2)], 1)


def test_counts_and_prf():
    gold = [(0, 2), (3, 4)]
    spans, ph = map_strings(TOK, ["Breast cancer", "tumour"])
    assert counts(gold, spans, ph) == (1, 1, 1)
    assert prf(1, 1, 1) == (0.5, 0.5, 0.5)
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)


def test_oracle_ceiling_unlabelled_repeat():
    # gold labels only the first "cancer": a perfect string list still yields one FP
    tags = ["O", "B-Disease", "O", "O", "O", "O", "O", "O", "O", "O"]
    gold = bio_spans(tags)
    spans, ph = map_strings(TOK, gold_strings(TOK, gold))
    assert counts(gold, spans, ph) == (1, 1, 0)
