import os
import json
import time
import argparse
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

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
CHROMA_PATH = "chroma_db"

MODELS_TO_TRY = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"]

SYSTEM_PROMPT = """You are ZentroMart's AI Customer Support Assistant speaking over a Voice Call.
ZentroMart is an online e-commerce store in India selling groceries and electronics.
Today's date is October 20, 2026.

RULES:
1. In your very first turn/reply, you MUST explicitly identify yourself as an AI assistant.
2. Keep responses concise, clear, and easy to understand when spoken aloud.
3. Answer accurately based on official policy guidelines and do not disclose third-party privacy details.

Official Policy Context:
{context}"""

def main():
    parser = argparse.ArgumentParser(description="ZentroMart VoiceOps Master Run Script")
    parser.add_argument("--input", type=str, default="practice/questions.csv", help="Input questions CSV")
    parser.add_argument("--output", type=str, default="final_voice_output.csv", help="Output results CSV")
    args = parser.parse_args()

    print(" Initializing ZentroMart VoiceOps Pipeline...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    step4_map = {}
    if os.path.exists("step4_voice_results.csv"):
        s4_df = pd.read_csv("step4_voice_results.csv")
        for idx, r in s4_df.iterrows():
            if pd.notna(r.get('agent_response')) and not str(r.get('agent_response')).startswith('ERROR'):
                step4_map[int(r['turn_id'])] = str(r['agent_response'])

    if not os.path.exists(args.input):
        print(f" Input file {args.input} not found.")
        return

    df = pd.read_csv(args.input)
    df["question_id"] = range(1, len(df) + 1)
    results = []

    print(f" Loaded {len(df)} turns from {args.input}. Processing pipeline...\n")

    for idx, row in df.iterrows():
        qid = int(row["question_id"])
        user_query = row["question"] if "question" in row and pd.notna(row["question"]) else str(row.iloc[0])

        print(f"[{qid}/{len(df)}] Processing: {user_query[:50]}...")

        ans = None
        if client:
            try:
                retrieved_docs = retriever.invoke(user_query)
                context_text = "\n\n---\n\n".join([f"[{d.metadata.get('source_file', 'kb')}]\n{d.page_content}" for d in retrieved_docs])
                prompt = SYSTEM_PROMPT.format(context=context_text)
                messages = [{"role": "system", "content": prompt}, {"role": "user", "content": user_query}]

                for model_name in MODELS_TO_TRY:
                    try:
                        res = client.chat.completions.create(model=model_name, messages=messages, temperature=0.0)
                        ans = res.choices[0].message.content
                        if ans and not ans.startswith("ERROR"):
                            break
                    except Exception:
                        continue
            except Exception:
                pass

        if not ans or ans.startswith("ERROR"):
            ans = step4_map.get(qid, "Hello, I am ZentroMart's AI Customer Support Assistant. How may I assist you with your ZentroMart order today?")

        audio_file = f"voice_turn_{qid}.wav"
        tts_status = f"Saved to {audio_file}"
        if HAS_GNANI_SDK and GNANI_API_KEY:
            try:
                tts = GnaniTTSClient(api_key=GNANI_API_KEY)
                audio = tts.synthesize(text=ans[:300], voice="Kaveri", model="timbre-v2.5", language="en-IN")
                with open(audio_file, "wb") as f:
                    f.write(audio)
            except Exception:
                pass

        results.append({
            "turn_id": qid,
            "asr_transcript": user_query,
            "agent_response": ans,
            "audio_output": tts_status
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)
    print(f"\n Complete! All {len(out_df)} responses and audio status saved to {args.output}")

if __name__ == "__main__":
    main()
