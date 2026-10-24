import os
import json
import time
import pandas as pd
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq

try:
    from gnani.tts import GnaniTTSClient
    HAS_GNANI_SDK = True
except ImportError:
    HAS_GNANI_SDK = False

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GNANI_API_KEY = os.getenv("GNANI_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

CHROMA_PATH = "chroma_db"
BATCH_SIZE = 10

# Only active fast models with available token limits
MODELS_TO_TRY = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768"
]

SYSTEM_PROMPT = """
You are ZentroMart's AI Customer Support Assistant speaking over a Voice Call.
ZentroMart is an online e-commerce store in India selling groceries and electronics.
Today's date is October 20, 2026.

RULES:
1. In your very first turn/reply, you MUST explicitly identify yourself as an AI assistant.
2. Keep responses concise, clear, and easy to understand when spoken aloud.
3. Answer accurately based on official policy guidelines and do not disclose third-party privacy details.

Official Policy Context:
{context}
"""

def generate_agent_response(user_query, retriever):
    retrieved_docs = retriever.invoke(user_query)
    context_text = "\n\n---\n\n".join([f"[{d.metadata.get('source_file', 'kb')}]\n{d.page_content}" for d in retrieved_docs])

    prompt = SYSTEM_PROMPT.format(context=context_text)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_query}
    ]

    for model_name in MODELS_TO_TRY:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str:
                time.sleep(1)
                continue
            else:
                continue

    return "ERROR: All model attempts failed."

def synthesize_voice_response(text, output_filename="sample_voice_output.wav"):
    if not HAS_GNANI_SDK or not GNANI_API_KEY:
        return "TTS Bypassed (Gnani Key or SDK missing)"

    try:
        tts = GnaniTTSClient(api_key=GNANI_API_KEY)
        audio = tts.synthesize(
            text=text[:300],
            voice="Kaveri",
            model="timbre-v2.5",
            language="en-IN"
        )
        with open(output_filename, "wb") as f:
            f.write(audio)
        return f"Saved to {output_filename}"
    except Exception as e:
        return f"TTS Error: {e}"

def evaluate_step4_voice():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    df = pd.read_csv("practice/questions.csv")
    df["question_id"] = range(1, len(df) + 1)

    output_csv = "step4_voice_results.csv"

    if os.path.exists(output_csv):
        try:
            results_df = pd.read_csv(output_csv)
            valid_mask = (
                pd.notna(results_df['agent_response']) & 
                (results_df['agent_response'].astype(str).str.strip() != '') & 
                (~results_df['agent_response'].astype(str).str.startswith('ERROR'))
            )
            # Filter out error rows so they get re-evaluated
            results_df = results_df[valid_mask]
            completed_ids = set(results_df['turn_id'])
        except Exception:
            results_df = pd.DataFrame()
            completed_ids = set()
    else:
        results_df = pd.DataFrame()
        completed_ids = set()

    total_valid = len(completed_ids)
    print(f" Step 4 Voice Progress: {total_valid}/52 valid responses completed.")

    if total_valid >= len(df):
        print(" All 52 turns are fully processed in Step 4 Voice Pipeline!")
        return

    pending_df = df[~df['question_id'].isin(completed_ids)].head(BATCH_SIZE)
    print(f" Processing pending turns: {pending_df['question_id'].tolist()}...\n")

    for idx, row in pending_df.iterrows():
        question_id = int(row["question_id"])
        user_query = row["question"] if "question" in row and pd.notna(row["question"]) else str(row.iloc[0])

        print(f"[{question_id}/52] Voice Processing: {user_query[:50]}...")

        agent_answer = generate_agent_response(user_query, retriever)

        audio_file = f"voice_turn_{question_id}.wav"
        tts_status = synthesize_voice_response(agent_answer, output_filename=audio_file)

        if 'turn_id' in results_df.columns:
            results_df = results_df[results_df['turn_id'] != question_id]

        new_row = pd.DataFrame([{
            "turn_id": question_id,
            "asr_transcript": user_query,
            "agent_response": agent_answer,
            "audio_output": tts_status
        }])

        results_df = pd.concat([results_df, new_row], ignore_index=True)
        results_df["turn_id"] = results_df["turn_id"].astype(int)
        
        try:
            results_df.sort_values("turn_id").to_csv(output_csv, index=False)
        except PermissionError:
            print("  Close CSV viewers to save progress!")

        time.sleep(1)

    completed_now = len(results_df[
        pd.notna(results_df['agent_response']) & 
        (~results_df['agent_response'].astype(str).str.startswith('ERROR'))
    ])
    print(f"\n Batch complete! Total clean progress: {completed_now}/52 saved to {output_csv}")

if __name__ == "__main__":
    evaluate_step4_voice()
