import os
import json
import logging
from google.cloud import storage, pubsub_v1
import vertexai
from vertexai.generative_models import GenerativeModel, Part

PROJECT_ID = os.environ.get("PROJECT_ID", "identity-res-e2e-10022026")
LOCATION = os.environ.get("LOCATION", "global")
OUTPUT_TOPIC = os.environ.get("OUTPUT_TOPIC", "raw-graph-events")
MODEL_NAME = "gemini-3.1-pro"

# Initialize clients once
storage_client = storage.Client(project=PROJECT_ID)
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, OUTPUT_TOPIC)

def download_from_gcs(bucket_name: str, file_name: str, local_path: str):
    """Downloads a file from GCS to a local path."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.download_to_filename(local_path)
    logging.info(f"Downloaded gs://{bucket_name}/{file_name} to {local_path}")

def build_extraction_prompt() -> str:
    return """
    You are an expert Data Extraction Agent for a Semantic Graph Platform.
    Your task is to analyze the provided multimodal document and extract entities and relationships.
    
    CRITICAL RULES:
    1. EXHAUSTIVE EXTRACTION: Extract relationships for ALL entities found in the document.
    2. Try to align extracted types with physical/material properties (e.g., Manufacturer, Material, TestResult).
    
    OUTPUT SCHEMA:
    You MUST output valid JSON conforming exactly to this structure:
    {
        "extraction_plan": "Step-by-step plan...",
        "extracted_nodes": [
            {"entity_name": "exact text from doc", "ontology_class": "suggested class", "properties": {"unit": "mls", "value": "10"}}
        ],
        "extracted_edges": [
            {"source_entity": "must match a node", "target_entity": "must match a node", "relationship_type": "type of relationship"}
        ]
    }
    """

def process_document(bucket_name: str, file_name: str):
    """Downloads the document, calls Gemini, and publishes the result."""
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    
    local_path = f"/tmp/{os.path.basename(file_name)}"
    download_from_gcs(bucket_name, file_name, local_path)
    
    import mimetypes
    mime_type, _ = mimetypes.guess_type(local_path)
    if not mime_type:
        mime_type = 'application/octet-stream'
        
    with open(local_path, "rb") as f:
        file_bytes = f.read()
        
    document_part = Part.from_data(data=file_bytes, mime_type=mime_type)
    
    logging.info(f"Executing Gemini 3.1 Pro extraction for {file_name}...")
    model = GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=[build_extraction_prompt()]
    )
    
    response = model.generate_content(
        [document_part],
        generation_config={"response_mime_type": "application/json", "temperature": 0.0}
    )
    
    try:
        parsed_response = json.loads(response.text)
        parsed_response["source_file"] = f"gs://{bucket_name}/{file_name}"
        
        # Publish to Pub/Sub
        message_bytes = json.dumps(parsed_response).encode("utf-8")
        future = publisher.publish(topic_path, data=message_bytes)
        message_id = future.result()
        logging.info(f"Successfully published extraction event to Pub/Sub. Message ID: {message_id}")
        
    except Exception as e:
        logging.error(f"Failed to parse or publish extraction result: {e}")
        raise
