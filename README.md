# BargainAI

> India's first Hinglish negotiation agent — MuRIL embeddings + hierarchical RAG + LLaMA 3.1 for real-time price bargaining.

## Problem Statement
Indian e-commerce and marketplace buyers naturally negotiate in Hinglish (mixed Hindi-English), but existing AI assistants are English-only and culturally tone-deaf to negotiation dynamics. A seller-side assistant that understands Hinglish context and responds with culturally appropriate counter-offers has no equivalent in the market.

## Architecture
User messages (Hinglish or English) are embedded using MuRIL (Google's multilingual BERT, optimized for Indian languages). A hierarchical RAG layer retrieves relevant product context and historical negotiation patterns from 14K product reviews (ChromaDB vector store). LLaMA 3.1 (via Groq API for low-latency inference) generates the negotiation response, conditioned on retrieved context + conversation history. FastAPI handles the backend; Gradio provides the chat interface.

## Tech Stack
`Python` · `MuRIL (Google)` · `LLaMA 3.1` · `Groq API` · `ChromaDB` · `LangChain` · `FastAPI` · `Gradio` · `HuggingFace Transformers`

## Key Results
- Multilingual: handles Hinglish, Hindi, and English inputs natively
- 14,000 product reviews indexed for retrieval context
- Sub-2s response latency via Groq inference
- First open-source Hinglish negotiation agent on HuggingFace

## Live Demo
🔗 [huggingface.co/spaces/nitz0219/bargainai](https://huggingface.co/spaces/nitz0219/bargainai)

## How to Run Locally
```bash
git clone https://github.com/niteshnankani-svg/bargainai
cd bargainai
python -m venv venv && source venv/bin/activate   # use Python 3.11
pip install -r requirements.txt
cp .env.example .env          # add GROQ_API_KEY
uvicorn app:app --reload
```
