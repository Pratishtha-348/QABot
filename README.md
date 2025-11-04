# 🤖 Advanced RAG QABot with RL Feedback

An **intelligent multi-tool Q&A assistant** built using **LangChain**, **Google Gemini**, and **Streamlit**, enhanced with **Reinforcement Learning (RL) feedback evaluation**.  
This system allows users to **upload documents or URLs**, chat with a **retrieval-augmented agent**, and receive **auto-evaluated and self-improving answers**.

---

## 🧭 Overview

This app combines **Retrieval-Augmented Generation (RAG)** with **agentic reasoning** and **reinforcement learning feedback** for continuous improvement.  
Users can:
- Upload PDFs or URLs to build custom knowledge bases  
- Chat naturally with an intelligent agent  
- Receive **quality-scored answers** with metrics on **relevance, length, structure, and confidence**  
- Edit past questions and regenerate answers with improved performance  

---

## 🧩 Key Features

| Feature | Description |
|----------|-------------|
| 🧠 **Multi-Tool Agent** | The agent uses 4 specialized tools: `RAGRetriever`, `Summarizer`, `Explainer`, and `Comparator`. |
| 📚 **Document & URL Upload** | Supports dynamic knowledge ingestion from PDFs or web content. |
| 📊 **RL-based Feedback** | Each answer is scored on `length`, `relevance`, `structure`, and `confidence`. |
| 🔄 **Auto-Refinement** | Low-scoring answers are automatically regenerated for higher quality. |
| ✏️ **Editable Questions** | Modify previous questions and regenerate improved responses. |
| 💾 **Persistent Chat History** | All chat sessions and metadata are saved via FastAPI backend. |
| 🎨 **Typing Effect + Metrics Visualization** | Smooth UI with live typing and detailed feedback. |

---

## 🏗️ System Architecture

      ┌─────────────────────┐
      │   Streamlit UI      │
      │ (app.py)            │
      │ - Upload Docs/URLs  │
      │ - Display Answers   │
      │ - Show Feedback     │
      └─────────┬───────────┘
                │ REST API
                ▼
      ┌─────────────────────┐
      │   FastAPI Backend   │
      │ /threadqa/chat      │
      │ - Stores sessions   │
      │ - Handles Q&A data  │
      └─────────┬───────────┘
                │
                ▼
      ┌─────────────────────┐
      │  LangChain Agent    │
      │ get_agentic_rag_*() │
      │ - RAG Retriever     │
      │ - Summarizer        │
      │ - Comparator        │
      │ - Explainer         │
      └─────────┬───────────┘
                │
                ▼
      ┌─────────────────────┐
      │   RL Feedback Loop  │
      │ agent_with_feedback │
      │ - Evaluates answers │
      │ - Assigns scores    │
      └─────────────────────┘

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/advanced-rag-qabot.git
cd advanced-rag-qabot
python -m venv venv
source venv/bin/activate    # For Linux/Mac
venv\Scripts\activate       # For Windows


3. Install dependencies
pip install -r requirements.txt

4. Set environment variables

Create a .env file in the root directory:

GOOGLE_API_KEY=your_gemini_api_key
FASTAPI_URL=http://localhost:8000

5. Run the FastAPI backend
uvicorn backend.main:app --reload --port 8000

6. Run the Streamlit frontend
streamlit run app.py

🧮 RL Feedback Mechanism

Each generated answer is automatically scored by the agent_with_feedback() module.
Metrics include:

Metric	Description
Length Score	Measures conciseness and completeness.
Relevance Score	Evaluates how closely the answer aligns with the query context.
Confidence Score	Based on model certainty or reasoning trace.
Structure Score	Evaluates formatting and logical organization.

If the overall score < 70%, the system refines the response automatically until improvement is achieved.

🧠 Usage

Launch the Streamlit app.

Click “➕ New Chat” in the sidebar.

Upload a PDF document or provide a URL.

Click “🚀 Build Agent” to initialize the agent.

Start chatting using the chat input box.

Optionally:

Edit previous questions to regenerate better answers

Expand metrics to view feedback scores

🧰 Tech Stack
Component	Technology
Frontend	Streamlit
Backend API	FastAPI
Core Logic	LangChain Agents
LLM	Google Gemini 1.5 Pro
Feedback Loop	Reinforcement Learning (custom evaluator)
Storage	JSON / Database (via FastAPI)
Deployment	Localhost / Cloud supported
🔍 Example Workflow

User uploads a document.

The system builds a RAG agent using LangChain retrievers.

User asks a question.

Agent retrieves context → generates answer → self-evaluates → refines.

Final answer is shown with performance metrics.

Chat history is saved persistently via FastAPI.
