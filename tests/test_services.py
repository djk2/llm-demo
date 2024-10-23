import os

import pytest

from llm_demo.services import DefaultLLMService, MockLLMService, get_llm_service_cls


def test_get_llm_service_cls():

    assert get_llm_service_cls("mock") == MockLLMService
    assert get_llm_service_cls("default") == DefaultLLMService

    with pytest.raises(ValueError):
        get_llm_service_cls(name="unknown")

    # Check the env variable works
    os.environ["LLM_SERVICE_CLASS"] = "default"
    assert get_llm_service_cls() == DefaultLLMService


def test_mock_llm_service_process_prompt():
    service = MockLLMService()
    prompt = ("What is John's phone number?",)
    text = (
        "John's phone number is <PHONE:12345678-9012-3456-7890-123456789012>. "
        "He lives at 123 Maple Street in San Francisco, "
        "and his email is john.doe@example.com."
    )

    assert service.process_prompt(prompt=prompt, text=text) == (
        "John's phone number is <PHONE:12345678-9012-3456-7890-123456789012>"
    )
