# schemas.py
from pydantic import BaseModel
from datetime import datetime

class ThreadQABase(BaseModel):
    session_id: str
    label: str
    question: str
    answer: str | None = None

class ThreadQACreate(ThreadQABase):
    pass

class ThreadQAResponse(ThreadQABase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True
