from fastapi import Depends, FastAPI

from llm_demo.dependencies import get_llm_service, get_tokenizer
from llm_demo.schemas import Payload
from llm_demo.services import BaseLLMService
from llm_demo.tokenizers import BaseTokenizer

app = FastAPI()


@app.post("/generate-answer")
async def generate_answer(
    payload: Payload,
    tokenizer: BaseTokenizer = Depends(get_tokenizer),
    llm_service: BaseLLMService = Depends(get_llm_service),
) -> dict:
    """
    Generate an answer for the given payload.
    """
    masked_set = tokenizer.mask(payload.context)
    llm_response = llm_service.process_prompt(prompt=payload.prompt, text=masked_set.text)
    return {"response": tokenizer.unmask(llm_response, masked_set.tokens)}
