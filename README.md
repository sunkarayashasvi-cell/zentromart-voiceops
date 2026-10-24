# ZentroMart VoiceOps 2026 — AI Voice Customer Support System

An end-to-end Voice-Based Customer Operations Agent built for **ZentroMart** (Indian E-Commerce store selling groceries & electronics) as part of the IIT Guwahati x Gnani.ai Capstone Project.

---

## 🚀 Key Features

* **Speech Recognition & Synthesis:** Integrated with Gnani.ai STT & TTS (	imbre-v2.5 / Kaveri).
* **Retrieval-Augmented Generation (RAG):** Date-aware policy grounding powered by ChromaDB vector store (kb/*.md).
* **Tool Calling & Database Lookup:** Dynamic order status tracking (get_order_details) and catalog lookup (check_sku_details).
* **Privacy & Boundary Guardrails:** Strictly complies with customer PII protection guidelines and explicit self-identification as an AI assistant.
* **Fallback & Rate-Limit Handling:** Multi-model fallback across Groq models (llama-3.1-8b-instant, llama-3.3-70b-versatile, mixtral-8x7b-32768).

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   `ash
   git clone [https://github.com/your-username/zentromart-voiceops.git](https://github.com/your-username/zentromart-voiceops.git)
   cd zentromart-voiceops
   `

2. **Install dependencies:**
   `ash
   pip install -r requirements.txt
   `

3. **Configure Environment Variables:**
   Copy .env.example to .env and fill in your API keys:
   `ash
   cp .env.example .env
   `

---

## 🏃 Execution Instructions

Run the unified master pipeline on the evaluation dataset:

`ash
python run.py --input practice/questions.csv --output final_voice_output.csv
`
