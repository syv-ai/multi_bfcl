"""Generation with a large language model."""

import os
import typing as t

import litellm
from litellm import Choices, ModelResponse

T = t.TypeVar("T")


def generate(
    prompt: str,
    model: str,
    api_base: str | None,
    temperature: float,
    response_format: type[T] | None = None,
) -> str | T:
    """Generate a response to a prompt.

    Args:
        prompt:
            The prompt to generate a response to.
        model:
            The model to use for generation.
        api_base:
            The base URL for the API, or None if no custom inference API is used.
        temperature:
            The temperature to use for generation.
        response_format (optional):
            The model to use for generation. If None then the response is returned as a
            string. Defaults to None.

    Returns:
        The generated response, which is a Pydantic model if `response_format` is set,
        and otherwise a string.

    Raises:
        RuntimeError:
            If the model failed to generate a response that fit the response format.
            This is only relevant if there is some post-initialisation validation
            happening in the response format.
    """
    conversation = [dict(role="user", content=prompt)]
    api_key = os.environ.get("OPENAI_API_KEY")
    auth_kwargs = (
        {"api_key": api_key or "dummy-api-key"} if api_base is not None else {}
    )

    response = litellm.completion(  # pyrefly: ignore[not-callable]
        model=model if api_base is None else f"openai/{model}",
        api_base=api_base,
        messages=conversation,
        temperature=temperature,
        response_format=response_format,
        timeout=600,
        **auth_kwargs,
    )
    assert isinstance(response, ModelResponse), (
        f"Expected a ModelResponse object, but got {type(response)}"
    )
    completion = response.choices[0].message.content
    assert completion is not None, f"The model did not return a completion: {response}"

    error_msgs: list[str] = list()
    if response_format is not None:
        for _ in range(num_attempts := 3):
            try:
                output = response_format.model_validate_json(completion)
                return output

            except Exception as e:
                error_msgs.append(str(e))

                conversation.extend(
                    [
                        dict(role="assistant", content=completion),
                        dict(role="user", content=str(e)),
                    ]
                )
                response = litellm.completion(  # pyrefly: ignore[not-callable]
                    model=model if api_base is None else f"openai/{model}",
                    api_base=api_base,
                    messages=conversation,
                    temperature=temperature,
                    response_format=response_format,
                    timeout=600,
                    **auth_kwargs,
                )
                choice = response.choices[0]
                assert isinstance(choice, Choices), (
                    f"Expected a Choices object, but got {type(choice)}"
                )
                completion = choice.message.content
                assert completion is not None, (
                    f"The model did not return a completion: {response}"
                )
        else:
            raise RuntimeError(
                f"Failed to validate the generated response after {num_attempts} "
                "attempts. Here is the final completion attempt:\n"
                f"{completion}\n\n"
                "Here are the errors that occurred:\n"
                f"{error_msgs}"
            )

    return completion
