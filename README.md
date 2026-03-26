#qa-bot
# Overview

This project is a Retrieval-Augmented Generation (RAG) application that allows users to upload a PDF and ask questions about its content.

# Features
Upload PDF documents
Extract and split text into chunks
Generate embeddings
Store in Chroma vector database
Retrieve relevant context
Answer questions using LLM
Simple Gradio UI

# Tech Stack
Python
LangChain
ChromaDB
Hugging Face / Watsonx (LLM)
Gradio

# Create environment
python3 -m venv my_env
source my_env/bin/activate

# Install dependencies
pip install -r requirements.txt
Run the App
python qabot.py

Then open: http://localhost:7860

# Environment Variables
export HUGGINGFACEHUB_API_TOKEN=your_token

Notes
Supports PDF files only
Uses similarity-based retrieval
Can be extended with chat history