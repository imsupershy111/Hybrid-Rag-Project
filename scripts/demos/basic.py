#!/usr/bin/env python3
"""
Hybrid RAG System - Demo Script
Main script demonstrating hybrid search with RAG using configuration file and document loader.
"""
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import yaml
from typing import Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI # OpenRouter dùng chuẩn của OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import EnsembleRetriever
from src.hybrid_rag.document_loader import DocumentLoaderUtility
from src.hybrid_rag.utils import configure_logging
from src.hybrid_rag.hybrid_retriever import create_document_type_aware_retriever
from src.hybrid_rag.query_preprocessor import QueryPreprocessor

def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    """Main function to run the hybrid RAG system."""

    # Configure logging to suppress warnings
    configure_logging()

    # Load configuration
    config = load_config()
    print("✅ Configuration loaded from config/config.yaml")

    # --- 1. Load Documents from Data Directory ---
    data_dir = config['data']['directory']
    data_path = Path(__file__).parent.parent.parent / data_dir
    loader = DocumentLoaderUtility(str(data_path), config=config)
    documents = loader.load_documents()

    if not documents:
        print("\n⚠️  No documents found in the data directory.")
        print(f"⚠️  Please add files to '{data_path}' directory.")
        print(f"⚠️  Supported formats: {', '.join(loader.get_supported_formats())}")
        return

    # --- 2. Initialize Models & Embeddings (Connecting to Ollama) ---
    embeddings = OllamaEmbeddings(model=config['ollama']['embedding_model'])
    llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    model=config['openrouter']['model'] # Nhớ thêm mục này vào config.yaml
    )       

    # --- 3. Create the Vector Store (Dense Search) ---
    persist_dir = Path(__file__).parent.parent.parent / config['vector_store']['persist_directory']

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print("♻️  Loading existing Vector Store từ ổ cứng...")
        vectorstore = Chroma(
            persist_directory=str(persist_dir), 
            embedding_function=embeddings
        )
    else:
        print("🏗️  Vector Store chưa có hoặc trống. Bắt đầu quá trình Embedding (sẽ mất thời gian)...")
        vectorstore = Chroma.from_documents(
            documents, 
            embeddings, 
            persist_directory=str(persist_dir)
        )
    
    vector_k = config['retrieval']['vector_search_k']
    print(f"✅ Vector Store Created with k={vector_k}.")

    # --- 4. Create Hybrid Retriever (CSV vs Text Aware) ---
    use_separate = config.get('document_processing', {}).get('use_separate_retrievers', False)
    # use_separate = False # Tạm thời tắt để tập trung vào demo cơ bản, có thể bật lại sau khi hoàn thiện phần retriever riêng cho CSV và Text
    if use_separate:
        print("🔧 Using document-type-aware retriever (CSV vs Text separation)")
        hybrid_retriever = create_document_type_aware_retriever(
            documents=documents,
            vectorstore=vectorstore,
            config=config
        )
    else:
        print("🔧 Using traditional ensemble retriever")
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": vector_k})

        keyword_retriever = BM25Retriever.from_documents(documents)
        keyword_k = config['retrieval']['keyword_search_k']
        keyword_retriever.k = keyword_k

        hybrid_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, keyword_retriever]
        )

    print("✅ Hybrid Retriever Created.")

    # --- 6. Define the Prompt (The Template Node) ---
    prompt = ChatPromptTemplate.from_template("""
    You are an expert assistant. Answer the user's question based ONLY on the provided context.
    If the context does not contain the answer, state clearly that the information is not available in the documents.

    <context>
    {context}
    </context>

    Question: {input}
    """)

    # --- 7. Construct the RAG Chain ---
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(hybrid_retriever, document_chain)
    print("✅ RAG Chain Constructed.")

    # --- 8. Execute Sample Queries ---
    sample_queries = [
        "Cho tôi biết tình trạng kho hàng (Status) và số lượng hiện có (Quantity On Hand) của sản phẩm có mã Specification là SPEC-000681."
    ]

    preprocessor = QueryPreprocessor()

    for query in sample_queries:
        print("\n" + "="*70)
        print(f"🔥 Executing Hybrid Query: '{query}'")
        print("="*70)

        expanded_query = preprocessor.expand_query(query)
        print(f"🚀 Expanded: {expanded_query}")

        try:
            response = rag_chain.invoke({"input": expanded_query})

            # --- 9. Output the Result ---
            print("\n--- Retrieved Context (Hybrid Results) ---")
            for i, doc in enumerate(response['context']):
                source = doc.metadata.get('source_file', 'unknown')
                content_preview = doc.page_content[:150].replace('\n', ' ')
                print(f"[{i+1}] Source: {source}")
                print(f"    {content_preview}...")

            print("\n--- Final LLM Answer ---")
            print(response['answer'])
            print("="*70)
        except Exception as e:
            print(f"❌ Error processing query: {e}")

    print("\n✅ Hybrid RAG demonstration complete!")


if __name__ == "__main__":
    main()