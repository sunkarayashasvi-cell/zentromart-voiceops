import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

SYSTEM_PROMPT = """
You are ZentroMart's AI Customer Support Assistant.
ZentroMart is an online e-commerce store in India selling groceries and electronics.

RULES:
1. In your very first turn/reply, you MUST explicitly identify yourself as an AI assistant.
2. Answer customer queries politely and professionally.
"""

BATCH_SIZE = 5

def get_active_groq_model():
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
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

def generate_with_retry(user_query, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=ACTIVE_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                print(f"  Rate limit hit on {ACTIVE_MODEL}. Retrying in 10s ({attempt}/{max_retries})...")
                time.sleep(10)
            else:
                print(f"  API Call failed: {e}")
                time.sleep(2)
    return f"ERROR: Failed after {max_retries} attempts."

def evaluate_step1_batch():
    df = pd.read_csv("practice/questions.csv")
    df["question_id"] = range(1, len(df) + 1)

    if os.path.exists("step1_baseline_results.csv"):
        try:
            results_df = pd.read_csv("step1_baseline_results.csv")
            if "question_id" in results_df.columns:
                results_df["question_id"] = pd.to_numeric(results_df["question_id"], errors="coerce").fillna(0).astype(int)
                
                # Exclude missing/blank/ERROR entries
                valid_mask = (
                    pd.notna(results_df['llm_answer']) & 
                    (results_df['llm_answer'].astype(str).str.strip() != '') & 
                    (results_df['llm_answer'].astype(str).str.strip() != 'nan') & 
                    (~results_df['llm_answer'].astype(str).str.startswith('ERROR'))
                )
                completed_ids = set(results_df[valid_mask]['question_id'])
            else:
                completed_ids = set()
        except Exception:
            results_df = pd.DataFrame(columns=["question_id", "query", "llm_answer"])
            completed_ids = set()
    else:
        results_df = pd.DataFrame(columns=["question_id", "query", "llm_answer"])
        completed_ids = set()

    total_completed = len(completed_ids)
    print(f" Completed so far: {total_completed}/52 questions.")

    if total_completed >= len(df):
        print(" All 52 questions are completely finished!")
        return

    # Select pending questions
    pending_df = df[~df['question_id'].isin(completed_ids)].head(BATCH_SIZE)
    print(f" Processing {len(pending_df)} pending question(s)...\n")

    for idx, row in pending_df.iterrows():
        question_id = int(row["question_id"])
        
        if "question" in row and pd.notna(row["question"]):
            user_query = row["question"]
        elif "asr_text" in row and pd.notna(row["asr_text"]):
            user_query = row["asr_text"]
        else:
            user_query = str(row.iloc[0])

        print(f"[{question_id}/52] Processing: {user_query[:45]}...")

        answer = generate_with_retry(user_query)

        results_df = results_df[results_df['question_id'] != question_id]
        new_row = pd.DataFrame([{"question_id": question_id, "query": user_query, "llm_answer": answer}])
        results_df = pd.concat([results_df, new_row], ignore_index=True)
        results_df["question_id"] = results_df["question_id"].astype(int)
        
        try:
            results_df.sort_values("question_id").to_csv("step1_baseline_results.csv", index=False)
        except PermissionError:
            print("  Close Excel or CSV viewers to save progress!")

        time.sleep(1)

    completed_now = len(results_df[
        pd.notna(results_df['llm_answer']) & 
        (results_df['llm_answer'].astype(str).str.strip() != '') & 
        (results_df['llm_answer'].astype(str).str.strip() != 'nan') & 
        (~results_df['llm_answer'].astype(str).str.startswith('ERROR'))
    ])
    print(f"\n Batch complete! Current valid progress: {completed_now}/52 saved to step1_baseline_results.csv")

if __name__ == "__main__":
    evaluate_step1_batch()
