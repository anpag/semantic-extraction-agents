# Extraction Agents (`extraction-agents`)

Production-grade LangGraph and Vertex AI Gemini document extraction service for the Neuro-Symbolic Knowledge Platform.

## Features
- **Dynamic Schema Retrieval**: Dynamically fetches domain ontology schema slices from Reasoning Engine REST API or BigQuery with fallback to built-in domain registry.
- **Multimodal PDF & Document Ingestion**: Ingests GCS PDF URIs, local PDFs, and text documents natively using Vertex AI Gemini multimodal capabilities.
- **Enriched Triple Typing**: Pydantic `Triple` model capturing `(subject, subject_class, predicate, object, object_class, confidence, unit, value, chunk_id, source_file)`.
- **Concurrent Chunk Extraction**: Asynchronous chunk processing with rate-limit exponential backoff and jitter retry via `tenacity`.
- **Zero Mocks & Hardcoded IDs**: Configurable via standard environment variables (`PROJECT_ID`, `LOCATION`, `MODEL_NAME`, `OUTPUT_TOPIC`, `REASONING_ENGINE_URL`, `BQ_DATASET`).

## Environment Variables
- `PROJECT_ID`: GCP Project ID.
- `LOCATION`: Vertex AI region (default: `us-central1`).
- `MODEL_NAME`: Gemini model identifier (e.g., `gemini-1.5-flash`, `gemini-1.5-pro`).
- `OUTPUT_TOPIC`: Pub/Sub output topic for extracted triples (default: `raw-graph-events`).
- `REASONING_ENGINE_URL`: Optional URL for Reasoning Engine REST API.
- `MAX_CONCURRENT_CHUNKS`: Max parallel chunks processed concurrently (default: `5`).

## Running Tests
```bash
pytest -v
```

