from sqlalchemy import Column, Integer, String, Text
from .database import Base

class ThreadQA(Base):
    __tablename__ = "thread_qa"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    label = Column(String, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
