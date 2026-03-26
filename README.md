# qa-bot-vr01
# Overview

This project is a Retrieval-Augmented Generation (RAG) application that allows users to upload a PDF and ask questions about its content.

# Features
Upload PDF documents <br>
Extract and split text into chunks <br>
Store in Chroma vector database <br>
Retrieve relevant context <br>
Answer questions using LLM <br>
Simple Gradio UI <br>

# Tech Stack
Python
LangChain
ChromaDB
Hugging Face / Watsonx (LLM)
Gradio

# Create environment
python3 -m venv my_env <br>
source my_env/bin/activate

# Install dependencies
pip install -r requirements.txt <br>
Run the App <br>
python qabot.py <br>

Then open: http://localhost:7860

# Environment Variables
export HUGGINGFACEHUB_API_TOKEN= "your_token"

# Notes
Supports PDF files only <br>
Uses similarity-based retrieval <br>
Can be extended with chat history <br>