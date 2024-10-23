from llm_demo.schemas import Payload


def test_api_generate_answer(client):

    payload = Payload(
        prompt="What is John's phone number?",
        context=(
            "John's phone number is 555-123-4567. "
            "He lives at 123 Maple Street in San "
            "Francisco, and his email is john.doe@example.com."
        ),
    )

    response = client.post("/generate-answer", json=payload.model_dump())
    assert response.status_code == 200
    assert response.json() == {"response": "John's phone number is 555-123-4567"}
