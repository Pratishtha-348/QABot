from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from . import models, schemas
from .database import engine, Base, get_db

# ---------------- CREATE TABLES ----------------
Base.metadata.create_all(bind=engine)

app = FastAPI(title="ThreadQA API")

# ---------------- CREATE Q&A (Insert a full message) ----------------
@app.post("/threadqa/chat/", response_model=schemas.ThreadQAResponse)
def create_chat_entry(data: schemas.ThreadQACreate, db: Session = Depends(get_db)):
    """
    Inserts a new Q&A row into the database with question + answer + label + session_id.
    Returns the inserted row.
    """
    qa = models.ThreadQA(**data.dict())
    db.add(qa)
    db.commit()
    db.refresh(qa)
    return qa  # ✅ Return the inserted row only

# ---------------- GET CHAT HISTORY ----------------
@app.get("/threadqa/chat/{session_id}", response_model=list[schemas.ThreadQAResponse])
def get_chat_history(session_id: str, db: Session = Depends(get_db)):
    """
    Fetches all Q&A entries for a given session_id.
    """
    qas = db.query(models.ThreadQA).filter(models.ThreadQA.session_id == session_id).all()
    if not qas:
        raise HTTPException(status_code=404, detail="No Q&A found for this session")
    return qas

# ---------------- UPDATE ANSWER ----------------
@app.put("/threadqa/chat/{qa_id}", response_model=schemas.ThreadQAResponse)
def update_chat_entry(qa_id: int, data: schemas.ThreadQACreate, db: Session = Depends(get_db)):
    """
    Updates an existing Q&A entry by ID.
    You can update the answer, question, or label.
    """
    qa = db.query(models.ThreadQA).filter(models.ThreadQA.id == qa_id).first()
    if not qa:
        raise HTTPException(status_code=404, detail="Q&A not found")
    for field, value in data.dict().items():
        setattr(qa, field, value)
    db.commit()
    db.refresh(qa)
    return qa

# ---------------- DELETE Q&A ----------------
@app.delete("/threadqa/chat/{qa_id}", response_model=dict)
def delete_chat_entry(qa_id: int, db: Session = Depends(get_db)):
    """
    Deletes a Q&A entry by ID.
    """
    qa = db.query(models.ThreadQA).filter(models.ThreadQA.id == qa_id).first()
    if not qa:
        raise HTTPException(status_code=404, detail="Q&A not found")
    db.delete(qa)
    db.commit()
    return {"detail": "Deleted successfully"}
