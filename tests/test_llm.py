"""Offline tests for LLM API-key handling."""

from unittest.mock import Mock

import pytest
from litellm import ModelResponse
from pydantic import BaseModel

from multi_bfcl.llm import generate


class _Answer(BaseModel):
    """A small structured response used to exercise repair handling."""

    value: int


def _response(content: str) -> ModelResponse:
    """Create a LiteLLM response without making a network request.

    Returns:
        A response object containing the supplied content.
    """
    return ModelResponse(
        choices=[{"message": {"content": content}, "index": 0, "finish_reason": "stop"}]
    )


def test_custom_endpoint_without_key_uses_placeholder_for_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both initial generation and structured repair carry the placeholder."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    completion = Mock(side_effect=[_response("not json"), _response('{"value": 42}')])
    monkeypatch.setattr("multi_bfcl.llm.litellm.completion", completion)

    result = generate(
        prompt="answer",
        model="model",
        api_base="https://example.invalid/v1",
        temperature=0,
        response_format=_Answer,
    )

    assert result == _Answer(value=42)
    assert completion.call_count == 2
    for call in completion.call_args_list:
        assert call.kwargs["api_key"] == "dummy-api-key"
        assert call.kwargs["api_base"] == "https://example.invalid/v1"


def test_custom_endpoint_preserves_real_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured OpenAI key is passed through unchanged."""
    monkeypatch.setenv("OPENAI_API_KEY", "real-secret")
    completion = Mock(return_value=_response("done"))
    monkeypatch.setattr("multi_bfcl.llm.litellm.completion", completion)

    assert (
        generate(
            prompt="answer",
            model="model",
            api_base="https://example.invalid/v1",
            temperature=0,
        )
        == "done"
    )
    assert completion.call_args.kwargs["api_key"] == "real-secret"


def test_native_provider_does_not_override_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native-provider auth remains delegated to LiteLLM's existing behaviour."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    completion = Mock(return_value=_response("done"))
    monkeypatch.setattr("multi_bfcl.llm.litellm.completion", completion)

    assert (
        generate(prompt="answer", model="native/model", api_base=None, temperature=0)
        == "done"
    )
    assert "api_key" not in completion.call_args.kwargs
    assert completion.call_args.kwargs["model"] == "native/model"
