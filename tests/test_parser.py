from __future__ import annotations

from cmg.parser import parse_turn


def test_no_annotations_yields_raw_text() -> None:
    text = "The bug is real because the loop terminates early."
    parsed = parse_turn(text)
    assert parsed.ops == ()
    assert parsed.visible_text == text
    assert parsed.parse_warnings == ()


def test_fence_block_extracted() -> None:
    text = (
        "Reasoning prefix.\n"
        "```cmg\n"
        '{"ops": [{"op": "commitment", "content": "bug is real", "refs": ["s-abc"]}]}\n'
        "```\n"
        "Final words."
    )
    parsed = parse_turn(text)
    assert len(parsed.ops) == 1
    assert parsed.ops[0]["op"] == "commitment"
    assert "```cmg" not in parsed.visible_text
    assert parsed.visible_text.startswith("Reasoning prefix.")
    assert parsed.visible_text.endswith("Final words.")


def test_xml_tag_extracted() -> None:
    text = (
        "The loop terminates early.\n"
        '<cmg ops=\'[{"op":"commitment","content":"real","refs":["s-x"]}]\'/>'
    )
    parsed = parse_turn(text)
    assert len(parsed.ops) == 1
    assert "<cmg" not in parsed.visible_text


def test_multiple_blocks_document_order() -> None:
    text = (
        "first.\n"
        "```cmg\n"
        '{"ops": [{"op": "support", "content": "a", "refs": []}]}\n'
        "```\n"
        "second.\n"
        '<cmg ops=\'[{"op":"commitment","content":"b","refs":["s-x"]}]\'/>\n'
        "third."
    )
    parsed = parse_turn(text)
    assert [o["op"] for o in parsed.ops] == ["support", "commitment"]


def test_malformed_json_does_not_raise() -> None:
    text = "```cmg\n{not valid json}\n```"
    parsed = parse_turn(text)
    assert parsed.ops == ()
    assert len(parsed.parse_warnings) == 1
    assert "invalid JSON" in parsed.parse_warnings[0]


def test_unexpected_shape_produces_warning() -> None:
    text = '```cmg\n"just a string"\n```'
    parsed = parse_turn(text)
    assert parsed.ops == ()
    assert any("expected list" in w for w in parsed.parse_warnings)


def test_non_dict_entries_are_dropped_silently() -> None:
    text = '```cmg\n{"ops": [1, "two", {"op": "support", "content": "x", "refs": []}]}\n```'
    parsed = parse_turn(text)
    assert len(parsed.ops) == 1


def test_mixed_valid_and_invalid_blocks() -> None:
    text = (
        "```cmg\n{not json}\n```\n"
        "```cmg\n"
        '{"ops": [{"op": "support", "content": "x", "refs": []}]}\n'
        "```"
    )
    parsed = parse_turn(text)
    assert len(parsed.ops) == 1
    assert len(parsed.parse_warnings) == 1


def test_oversize_input_is_truncated() -> None:
    from cmg.parser import MAX_PARSE_INPUT_BYTES

    huge = "x" * (MAX_PARSE_INPUT_BYTES + 1024)
    parsed = parse_turn(huge)
    assert any("truncated" in w for w in parsed.parse_warnings)


def test_nested_code_fence_outside_cmg() -> None:
    text = (
        "```python\nprint('hi')\n```\n"
        "```cmg\n"
        '{"ops": [{"op": "support", "content": "x", "refs": []}]}\n'
        "```"
    )
    parsed = parse_turn(text)
    assert len(parsed.ops) == 1
    assert "```python" in parsed.visible_text
