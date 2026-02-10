# Project Structure

```
.
├── .env
├── .env.example
├── Dockerfile
├── README.md
├── STRUCTURE.md
├── app/
│   ├── __init__.py
│   ├── api/
│   ├── app.py
│   ├── main.py
│   ├── schemas.py
│   └── streamlit_app.py
├── config/
│   ├── chunking_config.yaml
│   ├── logging_config.yaml
│   └── model_config.yaml
├── data/
│   ├── cache/
│   │   ├── .gitkeep
│   │   ├── a1_instructions_chunks.txt
│   │   └── retriever_graph.mmd
│   ├── embeddings/
│   │   └── .gitkeep
│   └── vectordb/
│       └── .gitkeep
├── docker-compose.yml
├── main.py
├── requirements.txt
├── scripts/
│   ├── build_embeddings.py
│   ├── cleanup.py
│   ├── run_tests.sh
│   └── setup_env.sh
├── src/
│   ├── __init__.py
│   ├── consultant/
│   │   ├── chroma_client.py
│   │   ├── crawler.py
│   │   ├── graph.py
│   │   ├── local_indexer.py
│   │   ├── prompts.py
│   │   └── retriever.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── base_llm.py
│   │   ├── gemini_client.py
│   │   ├── local_llm.py
│   │   ├── model_factory.py
│   │   └── pinecone_client.py
│   ├── graph/
│   │   ├── builder.py
│   │   ├── file_builder.py
│   │   ├── file_nodes.py
│   │   ├── file_state.py
│   │   ├── helpers.py
│   │   ├── main_graph.py
│   │   ├── nodes.py
│   │   └── state.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── inference_engine.py
│   │   └── response_parser.py
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── chunking.py
│   │   ├── drive_download.py
│   │   ├── pdf_forms.py
│   │   ├── pdf_to_text.py
│   │   ├── preprocessor.py
│   │   └── tokenizer.py
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── chain.py
│   │   └── templates.py
│   └── rag/
│       ├── __init__.py
│       ├── embedder.py
│       ├── indexer.py
│       ├── retriever.py
│       ├── retriever_graph.py
│       └── vector_store.py
├── tests/
│   ├── __init__.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_api_integration.py
│   │   └── test_end_to_end.py
│   └── unit/
│       ├── __init__.py
│       ├── test_chunking.py
│       ├── test_consultant_crawler.py
│       ├── test_consultant_local_indexer.py
│       ├── test_embedder.py
│       ├── test_file_endpoint.py
│       ├── test_indexer_single.py
│       ├── test_llm_clients.py
│       ├── test_pinecone_roundtrip.py
│       ├── test_prompts.py
│       ├── test_retriever_graph.py
│       ├── test_retriever_query_ltb_rules.py
│       └── test_retriever_three_indexes.py
```
