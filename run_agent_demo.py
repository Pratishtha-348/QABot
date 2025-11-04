# run_agent_demo.py
# --------------------------------------------------
# Interactive demo: load an existing PDF, index it with RAG,
# wrap it in an agent, ask questions, and auto-evaluate the answers.

import os
from chains.rag_agent import (
    get_agentic_rag_from_file,
    agent_with_feedback,
)

# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------
def main() -> None:
    print("🚀  Starting Agent Demo")

    # =======================================================================
    # Step 1 – Point to the document you want to use
    # =======================================================================
    print("\n1️⃣  Locating document…")
    
    # --- ⬇️ IMPORTANT: Make sure this path is correct! ⬇️ ---
    pdf_path = "/home/anviam124/AI-ML/Projects/QABot/data/Mastering RAG eBook.pdf"
    # -----------------------------------------------------------------

    # Check if the file actually exists before proceeding
    if not os.path.exists(pdf_path):
        print(f"❌ ERROR: The file was not found at the specified path: {pdf_path}")
        print("Please make sure the path is correct and the file is there.")
        return  # Exit if the file doesn't exist

    print(f"   ✅  Using document: {pdf_path}")


    # =======================================================================
    # Step 2 – Build the agent (this will vectorize your PDF)
    # =======================================================================
    print("\n2️⃣  Bootstrapping agent… (this can take a minute for a large PDF)")
    # Using the path to your existing PDF
    agent = get_agentic_rag_from_file(pdf_path, session_name="rag_ebook_session")
    print("   ✅  Agent ready!")


    # =======================================================================
    # Step 3 – Ask questions relevant to YOUR document
    # =======================================================================
    # ✅ FIXED: Questions are now relevant to a book about RAG
    questions = [
        "What are the core components of a RAG system?",
        "Explain the concept of 'chunking' for RAG.",
        "What are some common challenges when implementing RAG?",
    ]

    print("\n3️⃣  Querying…")
    for idx, q in enumerate(questions, start=1):
        print(f"\n📝  Question {idx}: {q}")
        
        # The agent_with_feedback function runs the agent and evaluates the answer
        answer, fb = agent_with_feedback(agent, q)
        
        print("\n🤖  Answer:")
        print(answer)
        print("\n📊  Feedback:", fb)
        print("".ljust(80, "═"))

    print("\n🎉  Demo complete!\n")


if __name__ == "__main__":
    main()