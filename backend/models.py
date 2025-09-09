from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class ThreadQA(Base):
    __tablename__ = "thread_qa"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    label = Column(String)
    question = Column(Text)
    answer = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)  # ✅ New field
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)  # ✅ New field
