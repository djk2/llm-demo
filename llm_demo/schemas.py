from pydantic import BaseModel


class Payload(BaseModel, from_attributes=True):
    prompt: str
    context: str
