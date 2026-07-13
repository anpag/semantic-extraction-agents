import os
import json
import vertexai
from vertexai.generative_models import GenerativeModel

# --- Configuration ---
PROJECT_ID = os.environ.get("PROJECT_ID", "semantic-graph-demo")
LOCATION = os.environ.get("LOCATION", "global") # Using global for 3.5-flash availability
MODEL_NAME = "gemini-3.5-flash"

def initialize_vertex():
    """Initializes the Vertex AI SDK."""
    print(f"Initializing Vertex AI in {PROJECT_ID}/{LOCATION} using {MODEL_NAME}...")
    vertexai.init(project=PROJECT_ID, location=LOCATION)

def resolve_semantic_anomaly(anomaly_type: str, raw_value: str, context: str):
    """
    Invokes the Semantic Resolution Agent to infer missing or unrecognized data.
    
    anomaly_type: "missing_unit" or "unrecognized_entity"
    raw_value: The incomplete or unrecognized string (e.g., "50", "caustic soda")
    context: The surrounding sentence/document text to aid inference.
    """
    initialize_vertex()
    
    system_instruction = f"""
    You are an expert Semantic Resolution Agent for a Clean Room Data Pipeline.
    Your job is to analyze incomplete or unrecognized data points and infer their canonical meaning based on the surrounding context.
    
    You are resolving an anomaly of type: {anomaly_type}
    
    RULES:
    1. If the anomaly is 'missing_unit', infer the most likely scientific unit of measurement based on the context (e.g., 'qudt:MilliL', 'qudt:Gram', 'qudt:DegC').
    2. If the anomaly is 'unrecognized_entity', infer the most likely canonical chemical or scientific name.
    3. You must provide a confidence score between 0.0 and 1.0. If you are not highly certain (>0.95), reflect that in the score so the system routes it to a human.
    
    OUTPUT SCHEMA:
    You MUST output valid JSON conforming exactly to this structure:
    {{
        "resolved_value": "the inferred canonical unit or name",
        "confidence_score": 0.98,
        "reasoning": "brief explanation of why this was chosen based on context"
    }}
    """
    
    model = GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=[system_instruction]
    )
    
    prompt = f"Raw Value: {raw_value}\nContext: {context}"
    
    print(f"Resolving anomaly: {raw_value} ({anomaly_type})...")
    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json", "temperature": 0.1}
    )
    
    return response.text

if __name__ == "__main__":
    # Test Case 1: Missing Unit
    context_text = "The reactor was pressurized to 10."
    anomaly_value = "10"
    anomaly_type = "missing_unit"
    
    print("--- Test Case 1: Missing Unit ---")
    try:
        result_json = resolve_semantic_anomaly(anomaly_type, anomaly_value, context_text)
        print(json.dumps(json.loads(result_json), indent=2))
    except Exception as e:
        print(f"Error: {e}")
        
    # Test Case 2: Unrecognized Synonym
    context_text = "The catalyst was dissolved in 50ml of IPA before addition."
    anomaly_value = "IPA"
    anomaly_type = "unrecognized_entity"
    
    print("\n--- Test Case 2: Unrecognized Synonym ---")
    try:
        result_json = resolve_semantic_anomaly(anomaly_type, anomaly_value, context_text)
        print(json.dumps(json.loads(result_json), indent=2))
    except Exception as e:
        print(f"Error: {e}")
