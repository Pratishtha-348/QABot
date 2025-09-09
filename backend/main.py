from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from . import models, schemas
from .database import engine, Base, get_db

# ---------------- CREATE TABLES ----------------
Base.metadata.create_all(bind=engine)

app = FastAPI(title="ThreadQA API")

# ---------------- CREATE Q&A ----------------
@app.post("/threadqa/chat/", response_model=schemas.ThreadQAResponse)
def create_chat_entry(data: schemas.ThreadQACreate, db: Session = Depends(get_db)):
    """
    Insert a new Q&A entry.
    """
    now = datetime.utcnow()
    qa = models.ThreadQA(
        **data.dict(),
        created_at=now,
        updated_at=now
    )
    db.add(qa)
    db.commit()
    db.refresh(qa)
    return qa

# ---------------- GET CHAT HISTORY ----------------
@app.get("/threadqa/chat/{session_id}", response_model=list[schemas.ThreadQAResponse])
def get_chat_history(session_id: str, db: Session = Depends(get_db)):
    """
    Fetch all Q&A entries for a session.
    """
    return db.query(models.ThreadQA).filter(models.ThreadQA.session_id == session_id).all()

# ---------------- UPDATE Q&A ----------------
@app.put("/threadqa/chat/{qa_id}", response_model=schemas.ThreadQAResponse)
def update_chat_entry(qa_id: int, data: schemas.ThreadQACreate, db: Session = Depends(get_db)):
    """
    Update question and/or answer of a Q&A entry.
    """
    qa = db.query(models.ThreadQA).filter(models.ThreadQA.id == qa_id).first()
    if not qa:
        raise HTTPException(status_code=404, detail="Q&A not found")

    for field, value in data.dict().items():
        setattr(qa, field, value)

    qa.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(qa)
    return qa

# ---------------- DELETE Q&A ----------------
@app.delete("/threadqa/chat/{qa_id}", response_model=dict)
def delete_chat_entry(qa_id: int, db: Session = Depends(get_db)):
    """
    Delete a Q&A entry by ID.
    """
    qa = db.query(models.ThreadQA).filter(models.ThreadQA.id == qa_id).first()
    if not qa:
        raise HTTPException(status_code=404, detail="Q&A not found")
    db.delete(qa)
    db.commit()
    return {"detail": "Deleted successfully"}
