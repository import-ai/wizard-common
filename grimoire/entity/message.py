from pydantic import BaseModel


class OpenAIMessage(BaseModel):
    role: str
    content: str


class Message(BaseModel):
    chunk_index: int | None = None
    start_index: int | None = None
    end_index: int | None = None
    conversation_id: str
    message_id: str
    message: OpenAIMessage
