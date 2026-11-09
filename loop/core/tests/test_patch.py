import pytest

from patch import apply_blocks, apply_text, apply_with_retries, extract_full_file, parse_blocks

SRC = "a = 1\nb = 2\n\ndef f(x):\n    return x + 1\n"


def block(search, replace):
    return "<<<<<<< SEARCH\n%s=======\n%s>>>>>>> REPLACE\n" % (search, replace)


def test_parse_single_block():
    assert parse_blocks(block("a = 1\n", "a = 2\n")) == [("a = 1\n", "a = 2\n")]


def test_parse_tolerates_prose_and_fences_and_keeps_order():
    text = (
        "Sure, here is the change.\n\n```python\n"
        + block("a = 1\n", "a = 2\n")
        + "```\n\nAnd a second one, because why not:\n\n```\n"
        + block("b = 2\n", "b = 3\nc = 4\n")
        + "```\nHope that helps!\n"
    )
    assert parse_blocks(text) == [("a = 1\n", "a = 2\n"), ("b = 2\n", "b = 3\nc = 4\n")]


def test_parse_tolerates_indented_markers_and_crlf():
    text = "   <<<<<<< SEARCH\r\na = 1\r\n   =======\r\na = 2\r\n   >>>>>>> REPLACE\r\n"
    assert parse_blocks(text) == [("a = 1\n", "a = 2\n")]


def test_parse_empty_replace_and_blank_line_are_distinct():
    assert parse_blocks(block("a = 1\n", "")) == [("a = 1\n", "")]
    assert parse_blocks(block("a = 1\n", "\n")) == [("a = 1\n", "\n")]


@pytest.mark.parametrize("text", [
    "<<<<<<< SEARCH\na = 1\n=======\na = 2\n",           # no terminator
    "<<<<<<< SEARCH\na = 1\n>>>>>>> REPLACE\n",          # no divider
    "<<<<<<< SEARCH\na = 1\n<<<<<<< SEARCH\n",           # nested
])
def test_parse_malformed_raises(text):
    with pytest.raises(ValueError):
        parse_blocks(text)


def test_no_blocks_in_plain_prose():
    assert parse_blocks("just talking\n=====\nno edits here\n") == []


def test_exact_match_applies():
    out, rep = apply_blocks(SRC, [("a = 1\n", "a = 10\n")])
    assert out == "a = 10\nb = 2\n\ndef f(x):\n    return x + 1\n"
    assert rep["ok"] and rep["mode"] == "search_replace" and rep["fallback_blocks"] == []


def test_multi_line_search_and_delete_and_insert():
    out, rep = apply_blocks(SRC, [("def f(x):\n    return x + 1\n", ""), ("b = 2\n", "b = 2\nc = 3\n")])
    assert rep["ok"]
    assert out == "a = 1\nb = 2\nc = 3\n\n"


def test_blocks_apply_sequentially_on_updated_text():
    out, rep = apply_blocks(SRC, [("a = 1\n", "a = 5\n"), ("a = 5\n", "a = 6\n")])
    assert rep["ok"] and out.startswith("a = 6\n")


def test_matches_whole_lines_not_substrings():
    out, rep = apply_blocks("xa = 1\n", [("a = 1\n", "a = 2\n")])
    assert not rep["ok"] and rep["reason"] == "no match" and out == "xa = 1\n"


def test_whitespace_tolerant_trailing_only():
    src = "a = 1   \nb = 2\t\nc = 3\r\n"
    out, rep = apply_blocks(src, [("a = 1\nb = 2\n", "a = 9\nb = 9\n")])
    assert rep["ok"] and rep["fallback_blocks"] == [0]
    assert out == "a = 9\nb = 9\nc = 3\r\n"


def test_leading_whitespace_is_never_fuzzy():
    src = "def f():\n    return 1\n"
    out, rep = apply_blocks(src, [("def f():\nreturn 1\n", "x\n")])
    assert not rep["ok"] and rep["reason"] == "no match" and rep["failed_block"] == 0 and out == src


def test_multiple_match_failure_returns_original_and_names_block():
    src = "x = 1\ny = 2\nx = 1\n"
    out, rep = apply_blocks(src, [("y = 2\n", "y = 3\n"), ("x = 1\n", "x = 2\n")])
    assert out == src
    assert not rep["ok"] and rep["failed_block"] == 1
    assert rep["reason"] == "multiple matches (2)"
    assert "block 2 of 2" in rep["message"]


def test_multiple_match_under_whitespace_fallback_also_fails():
    src = "x = 1 \nx = 1\t\n"
    out, rep = apply_blocks(src, [("x = 1\n", "x = 2\n")])
    assert out == src and not rep["ok"] and rep["reason"].startswith("multiple matches")


def test_exact_match_wins_over_ambiguous_whitespace_variants():
    src = "x = 1\nx = 1 \n"
    out, rep = apply_blocks(src, [("x = 1\n", "x = 2\n")])
    assert rep["ok"] and out == "x = 2\nx = 1 \n" and rep["fallback_blocks"] == []


def test_no_match_failure_names_first_failing_block_and_applies_nothing():
    out, rep = apply_blocks(SRC, [("a = 1\n", "a = 2\n"), ("zzz\n", "q\n")])
    assert out == SRC
    assert not rep["ok"] and rep["failed_block"] == 1 and rep["reason"] == "no match"


def test_empty_search_rejected_and_empty_block_list():
    out, rep = apply_blocks(SRC, [("", "x = 1\n")])
    assert out == SRC and not rep["ok"] and rep["reason"] == "empty SEARCH"
    out, rep = apply_blocks(SRC, [])
    assert out == SRC and not rep["ok"]


def test_source_without_trailing_newline_is_preserved():
    out, rep = apply_blocks("a = 1\nb = 2", [("b = 2\n", "b = 3\n")])
    assert rep["ok"] and out == "a = 1\nb = 3"


def test_apply_text_end_to_end_and_parse_error_report():
    out, rep = apply_text(SRC, "Here:\n```\n" + block("a = 1\n", "a = 7\n") + "```\n")
    assert rep["ok"] and out.startswith("a = 7\n")
    out, rep = apply_text(SRC, "<<<<<<< SEARCH\na = 1\n=======\na = 2\n")
    assert out == SRC and not rep["ok"] and "unterminated" in rep["reason"]


def _big_file(n):
    return "".join("x%d = %d\n" % (i, i) for i in range(n))


def test_full_file_mode(base):
    text = "Here is the whole file.\n```python\n" + base + "```\nDone.\n"
    assert len(base.splitlines()) > 400
    out, rep = apply_text(SRC, text)
    assert rep["ok"] and rep["mode"] == "full_file" and out == base


def test_full_file_requires_more_than_400_lines():
    body = _big_file(400)
    assert extract_full_file("```python\n" + body + "```\n") is None
    assert extract_full_file("```python\n" + _big_file(401) + "```\n") is not None


def test_full_file_rejects_two_python_blocks_truncated_fence_and_markers():
    big = _big_file(450)
    assert extract_full_file("```python\n%s```\n```python\n%s```\n" % (big, big)) is None
    assert extract_full_file("```python\n" + big) is None  # truncated reply
    assert extract_full_file("```bash\n" + big + "```\n") is None
    out, rep = apply_text(SRC, "```python\n" + big + "```\n" + block("a = 1\n", "a = 2\n"))
    assert rep["mode"] == "search_replace" and out.startswith("a = 2\n")


def test_full_file_ignores_non_python_blocks_beside_it():
    text = "```bash\nls\n```\n```python\n" + _big_file(450) + "```\n"
    assert extract_full_file(text) is not None


def test_apply_text_nothing_usable():
    out, rep = apply_text(SRC, "I think you should try a bigger model.")
    assert out == SRC and not rep["ok"] and rep["mode"] == "none"


# ---------------------------------------------------------------- apply_with_retries

def test_retries_success_has_class_none_and_empty_hint():
    out, rep = apply_with_retries(SRC, block("a = 1\n", "a = 2\n"))
    assert out.startswith("a = 2\n") and rep["ok"]
    assert rep["failure_class"] == "none" and rep["retry_hint"] == ""
    assert rep["mode"] == "search_replace" and rep["fallback_blocks"] == []  # apply_text's keys are kept


def test_retries_no_match_hint_has_first_two_lines_and_nearest_source_line():
    out, rep = apply_with_retries(SRC, block("b  = 2\nzzz = 9\nthird = 3\n", "q\n"))
    assert out == SRC and not rep["ok"]
    assert rep["failure_class"] == "no_match" and rep["failed_block"] == 0
    hint = rep["retry_hint"]
    assert "block 1" in hint and "'b  = 2'" in hint and "'zzz = 9'" in hint and "third = 3" not in hint
    assert "line 2" in hint and "'b = 2'" in hint  # difflib nearest match, with the source's own text


def test_retries_nearest_line_keeps_the_source_indentation():
    out, rep = apply_with_retries(SRC, block("return x + 2\n", "return x\n"))
    assert rep["failure_class"] == "no_match"
    assert "line 5" in rep["retry_hint"] and "'    return x + 1'" in rep["retry_hint"]


def test_retries_no_match_without_a_close_line_has_no_nearest_clause():
    out, rep = apply_with_retries(SRC, block("completely_unrelated_statement_here()\n", "x\n"))
    assert rep["failure_class"] == "no_match" and "Nearest" not in rep["retry_hint"]
    assert "completely_unrelated_statement_here()" in rep["retry_hint"]


def test_retries_failure_on_a_later_block_names_it_and_searches_the_updated_text():
    reply = block("a = 1\n", "a = 1\nnew_line = 5\n") + block("new_lin = 5\n", "x\n")
    out, rep = apply_with_retries(SRC, reply)
    assert out == SRC and rep["failure_class"] == "no_match" and rep["failed_block"] == 1
    assert "block 2" in rep["retry_hint"] and "'new_line = 5'" in rep["retry_hint"] and "line 2" in rep["retry_hint"]
    assert "earlier blocks" in rep["retry_hint"]


def test_retries_multi_match_hint_lists_line_numbers():
    src = "x = 1\ny = 2\nx = 1\nz = 3\nx = 1\n"
    out, rep = apply_with_retries(src, block("x = 1\n", "x = 2\n"))
    assert out == src and rep["failure_class"] == "multi_match" and rep["failed_block"] == 0
    assert "lines 1, 3, 5" in rep["retry_hint"] and "'x = 1'" in rep["retry_hint"] and "exactly once" in rep["retry_hint"]


def test_retries_multi_match_under_whitespace_fallback_is_also_classified():
    out, rep = apply_with_retries("x = 1 \nx = 1\t\n", block("x = 1\n", "x = 2\n"))
    assert rep["failure_class"] == "multi_match" and "lines 1, 2" in rep["retry_hint"]


@pytest.mark.parametrize("reply,needle", [
    ("<<<<<<< SEARCH\na = 1\n=======\na = 2\n", "unterminated"),
    ("<<<<<<< SEARCH\na = 1\n>>>>>>> REPLACE\n", "missing ======= divider"),
    (block("", "a = 2\n"), "empty SEARCH"),
    ("I would raise the learning rate a bit.", "no SEARCH/REPLACE blocks"),
    ("", "no SEARCH/REPLACE blocks"),
])
def test_retries_parse_class_for_unusable_replies(reply, needle):
    out, rep = apply_with_retries(SRC, reply)
    assert out == SRC and not rep["ok"] and rep["failure_class"] == "parse"
    assert needle in rep["retry_hint"] and "SEARCH" in rep["retry_hint"] and "REPLACE" in rep["retry_hint"]


def test_retries_syntax_class_returns_original_and_points_at_the_line():
    out, rep = apply_with_retries(SRC, block("b = 2\n", "b = (2\n"))
    assert out == SRC and not rep["ok"] and rep["failure_class"] == "syntax"
    assert "does not parse" in rep["retry_hint"] and "line 2" in rep["retry_hint"] and "'b = (2'" in rep["retry_hint"]
    assert rep["mode"] == "search_replace"


def test_retries_syntax_check_covers_full_file_replies(base):
    broken = base.replace("class MLP(nn.Module):", "class MLP(nn.Module)", 1)
    out, rep = apply_with_retries(SRC, "```python\n" + broken + "```\n")
    assert out == SRC and rep["failure_class"] == "syntax" and rep["mode"] == "full_file"
    out, rep = apply_with_retries(SRC, "```python\n" + base + "```\n")
    assert out == base and rep["failure_class"] == "none"


def test_retries_hint_lines_are_truncated():
    long = "q = " + "9" * 500
    out, rep = apply_with_retries(SRC, block(long + "\n", "x\n"))
    assert rep["failure_class"] == "no_match" and long not in rep["retry_hint"] and "..." in rep["retry_hint"]
    assert len(rep["retry_hint"]) < 700
