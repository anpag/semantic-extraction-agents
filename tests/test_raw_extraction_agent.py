import sys
import os
import json
import pytest
from unittest.mock import patch, MagicMock

# Add parent directory to path to import agent scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from raw_extraction_agent import extract_triples_from_file, build_system_prompt

@patch('builtins.open', new_callable=MagicMock)
@patch('raw_extraction_agent.bigquery.Client')
@patch('raw_extraction_agent.GenerativeModel')
@patch('raw_extraction_agent.initialize_vertex')
@patch('raw_extraction_agent.fetch_ontology_context')
def test_extract_triples_from_file(mock_fetch_context, mock_init_vertex, mock_generative_model, mock_bq_client, mock_open):
    # Mock the BigQuery context fetch
    mock_fetch_context.return_value = ("MOCKED ONTOLOGY CONTEXT", False)
    
    # Mock the Vertex AI model
    mock_model_instance = MagicMock()
    mock_generative_model.return_value = mock_model_instance
    
    # Mock the LLM JSON response
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "extracted_nodes": [{"entity_name": "TestEntity", "ontology_class": "TestClass", "raw_properties": {}}],
        "extracted_edges": [],
        "unbound_knowledge": []
    })
    mock_model_instance.generate_content.return_value = mock_response
    
    # Mock the file read
    mock_open.return_value.__enter__.return_value.read.return_value = b"dummy file content"
    
    # Call the extraction function
    result_json = extract_triples_from_file("dummy_file.pdf")
    result = json.loads(result_json)
    
    # Assertions
    assert len(result["extracted_nodes"]) == 1
    assert result["extracted_nodes"][0]["entity_name"] == "TestEntity"
    
    # Verify the LLM was called with the correct generation config
    mock_model_instance.generate_content.assert_called_once()
    args, kwargs = mock_model_instance.generate_content.call_args
    assert kwargs["generation_config"]["response_mime_type"] == "application/json"


def test_build_system_prompt():
    """Validates that the generated system prompt contains the correct schemas and context."""
    prompt = build_system_prompt("MOCK CONTEXT")
    
    # Check that it injects the dynamic BigQuery context
    assert "MOCK CONTEXT" in prompt
    
    # Check that the strict JSON schema is enforced in the prompt text
    assert "extracted_nodes" in prompt
    assert "extracted_edges" in prompt
    assert "unbound_knowledge" in prompt
    assert "OUTPUT SCHEMA:" in prompt
