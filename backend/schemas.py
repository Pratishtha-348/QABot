# schemas.py
from pydantic import BaseModel

class ThreadQABase(BaseModel):
    session_id: str
    label: str
    question: str
    answer: str | None = None

class ThreadQACreate(ThreadQABase):
    pass

class ThreadQAResponse(ThreadQABase):
    id: int

    class Config:
        orm_mode = True
