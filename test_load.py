from src.hybrid_rag.document_loader import DocumentLoaderUtility

# Giả sử bạn đã có file trong thư mục data
loader = DocumentLoaderUtility(data_directory="./data")
docs = loader.load_documents()

for doc in docs[:2]: # Xem thử 2 đoạn đầu tiên
    print(f"Source: {doc.metadata['source_file']}")
    print(f"Category: {doc.metadata['doc_category']}")
    print(f"Content: {doc.page_content[:100]}...")
    print("-" * 20)