# 🎙️ ZentroMart VoiceOps 2026 — AI Voice Customer Support System

An end-to-end, multi-stage **Voice-Based AI Customer Support Pipeline** built for **ZentroMart** (an Indian e-commerce platform for groceries & electronics) as part of the **IIT Guwahati x Gnani.ai Capstone Project**.

---

## 📌 Executive Summary

ZentroMart VoiceOps handles 52 multi-turn customer operations queries spanning **order tracking, return/refund policies, product specifications, and account management**. The system incorporates:
* **Speech Recognition & Synthesis:** Powered by Gnani.ai STT & TTS (`timbre-v2.5` / `Kaveri` voice).
* **Date-Aware RAG:** Vector search via ChromaDB using versioned document policies (`kb/*.md`) grounded for October 2026.
* **Dynamic Tool Calling:** Live SQLite lookups for catalog (`check_sku_details`) and order status tracking (`get_order_details`).
* **Customer Privacy Guardrails:** Automatic PII protection preventing unauthorized third-party disclosures.
* **100% Policy Grounding:** Enforces AI self-identification in Turn 1 and zero hallucination across all 52 benchmark evaluation turns.

---

## 🏗️ Architecture & Pipeline Flow

┌────────────────┐       ┌─────────────────┐       ┌──────────────────┐
│  Customer Audio│ ───►  │    Gnani STT    │ ───►  │ User Transcript  │
└────────────────┘       └─────────────────┘       └────────┬─────────┘
│
┌────────────────┐       ┌─────────────────┐                ▼
│ Synthesized WAV│ ◄───  │    Gnani TTS    │ ◄───  ┌──────────────────┐
│  Audio Output  │       │ (timbre-v2.5) │       │     Groq LLM     │
└────────────────┘       └─────────────────┘       │   (LLaMA-3.3)    │
└────────┬─────────┘
│
┌─────────────────────┴─────────────────────┐
▼                                           ▼
┌──────────────────┐                        ┌──────────────────┐
│  ChromaDB (RAG)  │                        │  SQLite Database │
│ Policy Documents │                        │ Orders & Catalog │
└──────────────────┘                        └──────────────────┘

---

## 🛠️ Key Features & Guardrails

1. **Explicit AI Self-Identification:** Strictly complies with Rule 1 by explicitly introducing itself as ZentroMart's AI Support Assistant in the first turn.
2. **Date-Aware Temporal Grounding:** Accurately calculates return and refund windows relative to October 2026 policy dates.
3. **Privacy & PII Protection:** Automatically detects and refuses requests asking for third-party personal information (e.g., phone numbers or delivery addresses).
4. **Fallback Resilience:** Multi-model fallback across Groq API endpoints (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b-32768`) ensuring zero downtime during API rate-limit spikes.

---

## 📊 Benchmark Evaluation Results

| Metric | Baseline Model | VoiceOps RAG System |
| :--- | :---: | :---: |
| **Total Evaluation Turns** | 52 | **52** |
| **Policy Accuracy** | 32.6% | **100.0%** |
| **AI Self-Identification (Turn 1)** | ❌ Failed | **✅ Passed** |
| **Privacy Violation Rate** | 12.5% | **0.0% (Protected)** |
| **Hallucination Rate** | 67.4% | **0.0%** |

---

## 📁 Repository Structure

zentromart-voiceops/
├── run.py                       # Unified master execution pipeline script
├── final_voice_output.csv       # Final evaluation outputs (52 turns + WAV paths)
├── requirements.txt             # Project dependencies
├── .env.example                 # Environment variables template
├── README.md                    # Project documentation
├── kb/                          # Knowledge base policy documents (Markdown)
├── data/                        # SQLite catalog & order databases
└── chroma_db/                   # Local ChromaDB vector database index


---

## 🏃 Installation & Execution

### 1. Clone Repository & Install Dependencies
```bash
git clone [https://github.com/sunkarayashasvi-cell/zentromart-voiceops.git](https://github.com/sunkarayashasvi-cell/zentromart-voiceops.git)
cd zentromart-voiceops
pip install -r requirements.txt
cp .env.example .env
python run.py
---

Evaluation Report: ZentroMart_VoiceOps_Evaluation_Report.pdf (4-page technical evaluation).

Demo Video: ZentroMart_VoiceOps_Demo.mp4 (1080p demo walkthrough under 2 minutes).
