"""Test the chat-template formatting logic without hitting the network."""

from unittest.mock import MagicMock

from src.data import SYSTEM_PROMPT, _format_row, build_inference_prompt


def test_format_row_produces_three_role_messages():
    row = {"id": "x", "dialogue": "A: hi\nB: hi", "summary": "they greet"}
    out = _format_row(row)
    assert [m["role"] for m in out["messages"]] == ["system", "user", "assistant"]
    assert out["messages"][0]["content"] == SYSTEM_PROMPT
    assert "A: hi" in out["messages"][1]["content"]
    assert out["messages"][2]["content"] == "they greet"


def test_build_inference_prompt_uses_chat_template():
    tok = MagicMock()
    tok.apply_chat_template.return_value = "<formatted>"
    result = build_inference_prompt("A: hi", tok)
    assert result == "<formatted>"
    args, kwargs = tok.apply_chat_template.call_args
    assert kwargs["add_generation_prompt"] is True
    # No assistant turn at inference time.
    messages = args[0]
    assert [m["role"] for m in messages] == ["system", "user"]
