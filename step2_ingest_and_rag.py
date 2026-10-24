import os
import glob
import time
import requests
import pandas as pd
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

KB_DIR = "kb"
CHROMA_PATH = "chroma_db"
BATCH_SIZE = 5

SYSTEM_PROMPT = """
You are ZentroMart's AI Customer Support Assistant.
ZentroMart is an online e-commerce store in India selling ONLY groceries and electronics.

RULES:
1. In your very first turn/reply, you MUST explicitly identify yourself as an AI assistant.
2. Answer customer queries politely, professionally, and strictly based on the provided Official Policy Context below.
3. If the context does not contain enough information to answer, state clearly what policy rules apply or what information is missing.
4. Do NOT make up return policies, warranty terms, or product offerings outside of the context.

Official Policy Context:
{context}
"""

def get_active_groq_model():
    """Queries Groq API to discover active available models for this API key."""
    try:
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        resp = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=10)
        if resp.status_code == 200:
            models_data = resp.json().get("data", [])
            active_ids = [m["id"] for m in models_data]
            preferences = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-20b"]
            for pref in preferences:
                if pref in active_ids:
                    print(f" Selected active Groq model: '{pref}'")
                    return pref
            if active_ids:
                print(f" Selected available Groq model: '{active_ids[0]}'")
                return active_ids[0]
    except Exception as e:
        print(f"  Could not fetch models dynamically: {e}")
    return "openai/gpt-oss-20b"

ACTIVE_MODEL = get_active_groq_model()

def get_latest_kb_files():
    """Resolves versioned markdown files in kb/ ensuring highest version takes precedence."""
    all_files = glob.glob(os.path.join(KB_DIR, "*.md"))
    version_groups = {}

    for filepath in all_files:
        filename = os.path.basename(filepath)
        # Check for version pattern like pol_returns_v4.md
        if "_v" in filename:
            base_prefix = filename.split("_v")[0]
            try:
                ver_num = int(filename.split("_v")[1].replace(".md", ""))
            except ValueError:
                ver_num = 0
            
            if base_prefix not in version_groups or ver_num > version_groups[base_prefix][1]:
                version_groups[base_prefix] = (filepath, ver_num)
        else:
            base_prefix = filename.replace(".md", "")
            if base_prefix not in version_groups:
                version_groups[base_prefix] = (filepath, 0)

    selected_files = [item[0] for item in version_groups.values()]
    print(f" Identified {len(selected_files)} active Markdown policy files from '{KB_DIR}/'.")
    return selected_files

def build_vector_store():
    """Ingests Markdown files, creates text chunks, and builds/loads local ChromaDB embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        print(f" Loading existing ChromaDB vector store from '{CHROMA_PATH}'...")
        vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        return vectorstore

    print(" Ingesting and embedding Knowledge Base documents...")
    files = get_latest_kb_files()
    documents = []

    for fpath in files:
        loader = TextLoader(fpath, encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = os.path.basename(fpath)
        documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f" Created {len(chunks)} text chunks from KB policy files.")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f" Vector store successfully built and saved to '{CHROMA_PATH}'.")
    return vectorstore

def generate_rag_answer(user_query, retriever, max_retries=3):
    """Retrieves top relevant policy chunks and generates a grounded response."""
    retrieved_docs = retriever.invoke(user_query)
    context_text = "\n\n---\n\n".join([f"[{d.metadata.get('source_file', 'kb')}]\n{d.page_content}" for d in retrieved_docs])

    prompt = SYSTEM_PROMPT.format(context=context_text)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=ACTIVE_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                print(f"  Rate limit hit. Waiting 10s (Attempt {attempt}/{max_retries})...")
                time.sleep(10)
            else:
                print(f"  API error: {e}")
                time.sleep(2)
    return f"ERROR: Failed after {max_retries} attempts."

def evaluate_step2_batch():
    vectorstore = build_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    df = pd.read_csv("practice/questions.csv")
    df["question_id"] = range(1, len(df) + 1)

    output_csv = "step2_rag_results.csv"

    if os.path.exists(output_csv):
        try:
            results_df = pd.read_csv(output_csv)
            if "question_id" in results_df.columns:
                results_df["question_id"] = pd.to_numeric(results_df["question_id"], errors="coerce").fillna(0).astype(int)
                valid_mask = (
                    pd.notna(results_df['rag_answer']) & 
                    (results_df['rag_answer'].astype(str).str.strip() != '') & 
                    (results_df['rag_answer'].astype(str).str.strip() != 'nan') & 
                    (~results_df['rag_answer'].astype(str).str.startswith('ERROR'))
                )
                completed_ids = set(results_df[valid_mask]['question_id'])
            else:
                completed_ids = set()
        except Exception:
            results_df = pd.DataFrame(columns=["question_id", "query", "rag_answer"])
            completed_ids = set()
    else:
        results_df = pd.DataFrame(columns=["question_id", "query", "rag_answer"])
        completed_ids = set()

    total_completed = len(completed_ids)
    print(f" RAG Evaluation Progress: {total_completed}/52 questions completed.")

    if total_completed >= len(df):
        print(" All 52 questions are fully answered with RAG!")
        return

    pending_df = df[~df['question_id'].isin(completed_ids)].head(BATCH_SIZE)
    print(f" Processing batch of {len(pending_df)} questions with RAG...\n")

    for idx, row in pending_df.iterrows():
        question_id = int(row["question_id"])
        
        if "question" in row and pd.notna(row["question"]):
            user_query = row["question"]
        elif "asr_text" in row and pd.notna(row["asr_text"]):
            user_query = row["asr_text"]
        else:
            user_query = str(row.iloc[0])

        print(f"[{question_id}/52] RAG Processing: {user_query[:45]}...")

        answer = generate_rag_answer(user_query, retriever)

        results_df = results_df[results_df['question_id'] != question_id]
        new_row = pd.DataFrame([{"question_id": question_id, "query": user_query, "rag_answer": answer}])
        results_df = pd.concat([results_df, new_row], ignore_index=True)
        results_df["question_id"] = results_df["question_id"].astype(int)
        
        try:
            results_df.sort_values("question_id").to_csv(output_csv, index=False)
        except PermissionError:
            print(" Close Excel or CSV viewers to allow saving!")

        time.sleep(1)

    completed_now = len(results_df[
        pd.notna(results_df['rag_answer']) & 
        (results_df['rag_answer'].astype(str).str.strip() != '') & 
        (results_df['rag_answer'].astype(str).str.strip() != 'nan') & 
        (~results_df['rag_answer'].astype(str).str.startswith('ERROR'))
    ])
    print(f"\n Batch complete! Current RAG progress: {completed_now}/52 saved to {output_csv}")

if __name__ == "__main__":
    evaluate_step2_batch()
