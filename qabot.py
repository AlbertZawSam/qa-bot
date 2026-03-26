import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

import gradio as gr


def warn(*args, **kwargs):
    pass


import warnings
warnings.warn = warn
warnings.filterwarnings("ignore")

# LLM
def get_llm():
    model_id = os.getenv("HF_LLM_MODEL", "google/flan-t5-small")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    generation_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.5,
    )
    return HuggingFacePipeline(pipeline=generation_pipeline)

# Document loader
def document_loader(file):
    """Accepts either a Gradio file object or a filepath string."""
    file_path = getattr(file, "name", file)
    loader = PyPDFLoader(file_path)
    loaded_document = loader.load()
    return loaded_document

# Text splitter
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks


## Embedding model
def local_embedding():
    embedding_model = HuggingFaceEmbeddings(
        model_name=os.getenv(
            "HF_EMBED_MODEL",
            "sentence-transformers/paraphrase-MiniLM-L3-v2",
        )
    )
    return embedding_model

## Vector db
def vector_database(chunks):
    embedding_model = local_embedding()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb

## Retriever
def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever()
    return retriever

# QA Chain
def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=True,
    )

    response = qa.invoke({"query": query})
    return response["result"]

# Gradio interface
rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Output"),
    title="RAG-based PDF QA Bot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)

def env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main():
    port = os.getenv("GRADIO_SERVER_PORT")
    rag_application.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(port) if port else None,
        share=env_flag("GRADIO_SHARE", True),
    )


if __name__ == "__main__":
    main()
