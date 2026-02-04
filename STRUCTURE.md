# Project Structure

```
.
├── Dockerfile
├── README.md
├── STRUCTURE.md
├── app/
│   ├── __init__.py
│   ├── api/
│   └── main.py
├── config/
│   ├── logging_config.yaml
│   └── model_config.yaml
├── data/
│   ├── cache/
│   │   └── .gitkeep
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
│   ├── core/
│   │   ├── __init__.py
│   │   ├── base_llm.py
│   │   ├── gemini_client.py
│   │   ├── local_llm.py
│   │   └── model_factory.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── inference_engine.py
│   │   └── response_parser.py
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── chunking.py
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
│       └── vector_store.py
├── tests/
│   ├── __init__.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_api_integration.py
│   │   └── test_end_to_end.py
│   └── unit/
│       ├── __init__.py
│       ├── test_llm_clients.py
│       └── test_prompts.py
├── .env
```
