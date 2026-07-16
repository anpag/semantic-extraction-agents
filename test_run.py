import os
import json
from graph import app

def test():
    # Injecting the text directly into document_uri for local testing purposes
    # so Gemini receives the text in the prompt without needing GCS access.
    dummy_text = """
    Document Title: Analysis of Project HotDump Gloo.
    Page 1: Introduction. This report covers the Formulation of a new polymer blend.
    Page 2: Methods. We conducted a LapShearTest on the material using an Instron machine. The result was 45 MPa.
    Page 3: Conclusion. The material meets the criteria.
    """
    
    initial_state = {
        "bucket_name": "mock-bucket",
        "file_name": "mock.pdf",
        "document_uri": dummy_text,
        "primary_classes": [],
        "chunks": [],
        "extracted_triples": [],
        "errors": []
    }
    
    print("Starting LangGraph Orchestrator Test...")
    final_state = app.invoke(initial_state)
    
    print("\n--- FINAL GRAPH STATE ---")
    print(json.dumps(final_state, indent=2, default=str))

if __name__ == "__main__":
    test()
