# Hybrid RAG & MCP

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Introduction
This project is an advanced Hybrid Retrieval-Augmented Generation (RAG) system designed for intelligent querying across both structured (CSV) and unstructured (Markdown, Text) data. The highlight of this project is its native integration with Claude Desktop via the Model Context Protocol (MCP).
---

## Overview

This project implements a hybrid RAG system that combines:
- **Semantic Search**: Dense vector embeddings for understanding meaning and context
- **Keyword Search**: BM25 sparse retrieval for exact keyword matching
- **Hybrid Fusion**: Reciprocal Rank Fusion (RRF) to combine results from both methods
- **MCP Server**: Model Context Protocol server for Claude integration
- **Multi-format Support**: Automatically loads documents from various file formats

The hybrid approach ensures better retrieval accuracy by leveraging the strengths of both search methods.

## Features

- Vector-based semantic search using Chroma and Ollama embeddings
- BM25 keyword search for exact term matching
- Ensemble retriever with Reciprocal Rank Fusion (RRF)
- Support for multiple document formats (TXT, PDF, MD, DOCX, CSV)
- Automated document loading from data directory
- Model Context Protocol (MCP) server for Claude Desktop/API integration
- Hardware Optimized: Specifically tuned for 8GB RAM environments using a dual-model setup (Local Embeddings + Cloud LLM via OpenRouter).

## Architecture

```
User Documents → data/ directory
                      ↓
            Document Loader
                      ↓
Query → Hybrid Retriever → [Vector Retriever + BM25 Retriever]
                         → RRF Fusion
                         → Retrieved Context
                         → LLM (Ollama)
                         → Final Answer
```

## Prerequisites

1. **Python 3.9+**
2. **Ollama** Installed and running with the nomic-embed-text model
3. OpenRouter API Key: To access high-performance LLMs without local hardware strain

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd hybrid-rag-project
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
Hybrid-Rag-Project/
├── src/
│   └── hybrid_rag/           # Core package (Loaders, Retrievers, Preprocessors)
├── scripts/
│   ├── mcp/                  # MCP Server implementations for Claude
│   └── demos/                # Command-line interface demos (Basic & Conversational)
├── config/                   # YAML configurations and MCP JSON templates
├── data/                     # Data directory (CSV, MD, TXT files)
├── chroma_db/                # Local vector database (auto-generated)
└── README.md                 # Project documentation
```

## Sample Data (UCSC Extension Project)

This repository includes **13 sample data files** for demonstration and testing purposes. These files represent a realistic business scenario for TechVision Electronics and are designed to showcase the system's capabilities across multiple document types.

### 📊 Included Sample Files

**Structured Data (CSV) - 7 files:**
- `product_catalog.csv` - Product inventory with specifications (5,000 rows)
- `inventory_levels.csv` - Stock levels and warehouse data (10,000 rows)
- `sales_orders_november.csv` - Monthly sales transactions (8,000 rows)
- `warranty_claims_q4.csv` - Customer warranty claims (3,000 rows)
- `production_schedule_dec2024.csv` - Manufacturing schedule (4,000 rows)
- `supplier_pricing.csv` - Vendor pricing information (6,000 rows)
- `shipping_manifests.csv` - Shipping and logistics data (5,000 rows)

**Unstructured Data (Markdown) - 5 files:**
- `customer_feedback_q4_2024.md` - Customer reviews and feedback (600 chunks)
- `market_analysis_2024.md` - Market research and trends (400 chunks)
- `quality_control_report_nov2024.md` - QC findings and issues (501 chunks)
- `return_policy_procedures.md` - Policy documentation (300 chunks)
- `support_tickets_summary.md` - Technical support summary (700 chunks)

**Text Data - 1 file:**
- `product_specifications.txt` - Technical specifications (334 chunks)

**Total Dataset:**
- **41,000 CSV rows** (chunked into 41,000 documents at 10 rows per chunk)
- **2,835 text/markdown chunks** (chunked at 1000 chars with 200 char overlap)
- **43,835 total searchable document chunks**

### 🎯 Purpose

These sample files are included to:
1. **Demonstrate** the system's hybrid search capabilities
2. **Test** both semantic (vector) and lexical (keyword) retrieval
3. **Validate** document-type-aware retrieval architecture
4. **Provide** immediate working examples without additional setup
5. **Showcase** cross-document query synthesis

**For Production Use:**
To use your own data instead:
1. Remove or backup the sample files from `data/`
2. Add your own documents (TXT, PDF, MD, DOCX, CSV)
3. Re-run ingestion
4. Optionally uncomment data exclusions in `.gitignore

Modify this file to:
- Use different LLM models from OpenRouter
- Change the data directory location
- Adjust retrieval parameters (k values)
- Change vector store persistence location

## Usage

### Claude Desktop/API via MCP

The MCP (Model Context Protocol) server allows Claude to directly query your local RAG system.

#### Setup for Claude Desktop

1. **First, add documents to your data directory**:
```bash
cp /path/to/your/documents/*.pdf data/
```

2. **Edit the `config/claude_desktop_config.json` file** to use the correct absolute path:
```json
{
  "mcpServers": {
    "hybrid-rag": {
      "command": "python",
      "args": [
        "/absolute/path/to/hybrid-rag-project/scripts/mcp_server_claude.py"
      ],
      "env": {
        "PYTHONPATH": "/absolute/path/to/hybrid-rag-project"
      }
    }
  }
}
```

3. **Add this configuration to Claude Desktop**:

   **On macOS**:
   ```bash
   # Copy the configuration
   mkdir -p ~/Library/Application\ Support/Claude
   # Edit the file and add your MCP server configuration
   nano ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```

   **On Windows**:
   ```
   %APPDATA%\Claude\claude_desktop_config.json
   ```

   **On Linux**:
   ```
   ~/.config/Claude/claude_desktop_config.json
   ```

4. **Restart Claude Desktop**

5. **In Claude Desktop, you'll now see the MCP tools available**. You can ask Claude:
   - "Use the ingest_documents tool to load my documents"
   - "Query my documents about [your question]"
   - "Check the status of the RAG system"

#### Available MCP Tools

Claude will have access to these tools:

**Document Ingestion & Search:**
- **`ingest_documents`**: Start loading and indexing documents asynchronously from the data/ directory
- **`get_ingestion_status`**: Monitor the progress of document ingestion (percentage, current file, stage)
- **`query_documents`**: Query the documents using hybrid search (semantic + keyword)
- **`get_status`**: Check the RAG system status

**Structured Data Queries (for CSV files):**
- **`list_datasets`**: List all available CSV datasets with columns and row counts
- **`count_by_field`**: Count rows where a field matches a value (e.g., "count people named Michael")
- **`filter_dataset`**: Get all rows matching field criteria (e.g., "all people from Company X")
- **`get_dataset_stats`**: Get statistics about a dataset (rows, columns, memory usage)

#### Async Ingestion with Progress Tracking

The ingestion process now runs asynchronously with real-time progress updates:

- **Non-blocking**: Ingestion runs in the background
- **Progress tracking**: See percentage complete (0-100%)
- **File-level updates**: Know which file is currently being processed
- **Stage information**: Loading files (0-80%) → Building index (80-100%) → Completed
- **Status monitoring**: Check progress at any time with `get_ingestion_status`

**When to use each approach:**
- **Structured queries** (`count_by_field`, `filter_dataset`): For exact counts, filtering, and structured data
- **Semantic search** (`query_documents`): For conceptual questions, understanding content, summarization

## Supported File Formats

The system automatically loads and processes these formats:
- `.txt` - Plain text files
- `.pdf` - PDF documents
- `.md` - Markdown files
- `.docx` - Microsoft Word documents
- `.csv` - CSV files

Simply drop any supported files into the `data/` directory!

## How It Works

### Document Loading

The `DocumentLoaderUtility` class:
1. Scans the `data/` directory recursively
2. Identifies supported file formats
3. Uses appropriate loaders for each format
4. Adds metadata (source file, file type) to each document
5. Returns a list of `Document` objects ready for indexing

### Hybrid Retrieval

The `EnsembleRetriever` uses Reciprocal Rank Fusion (RRF) to:
1. Retrieve top-k results from vector search (semantic)
2. Retrieve top-k results from BM25 search (keyword)
3. Assign reciprocal rank scores to each result
4. Combine scores to produce a unified ranking
5. Return the most relevant documents overall

This approach handles:
- Semantic queries ("How do I request time off?")
- Keyword queries ("PTO form HR-42")
- Complex queries benefiting from both methods

## Dependencies

Core libraries:
- `langchain`: Framework for LLM applications
- `langchain-community`: Community integrations
- `langchain-ollama`: Ollama integration
- `chromadb`: Vector database for embeddings
- `rank-bm25`: BM25 implementation for keyword search
- `fastapi`: Web framework for API
- `uvicorn`: ASGI server
- `pyyaml`: YAML configuration parsing

Document loaders:
- `pypdf`: PDF processing
- `python-docx`: Word document processing
- `unstructured`: Markdown and other formats

## License

This project is provided as-is for educational and demonstration purposes.

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Ollama Documentation](https://ollama.ai/docs)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [End to end Hybrid Rag Project](https://github.com/gwyer/hybrid-rag-project)
