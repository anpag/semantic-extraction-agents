import sys
import os
import json
import pytest
from unittest.mock import patch, MagicMock

# Add parent directory to path to import agent scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from semantic_resolution_agent import resolve_semantic_anomaly

@patch('semantic_resolution_agent.GenerativeModel')
@patch('semantic_resolution_agent.initialize_vertex')
def test_resolve_semantic_anomaly(mock_init_vertex, mock_generative_model):
    # Mock the Vertex AI model instance
    mock_model_instance = MagicMock()
    mock_generative_model.return_value = mock_model_instance
    
    # Mock the model's response
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "resolved_value": "qudt:BAR",
        "confidence_score": 0.75,
        "reasoning": "Test reasoning based on context."
    })
    mock_model_instance.generate_content.return_value = mock_response
    
    # Call the function being tested
    result_json = resolve_semantic_anomaly("missing_unit", "10", "The reactor was pressurized to 10.")
    result = json.loads(result_json)
    
    # Assertions
    assert result["resolved_value"] == "qudt:BAR"
    assert result["confidence_score"] == 0.75
    
    # Verify the model was called with the correct prompt and config
    mock_model_instance.generate_content.assert_called_once()
    args, kwargs = mock_model_instance.generate_content.call_args
    assert "Raw Value: 10" in args[0]
    assert "Context: The reactor was pressurized to 10." in args[0]
    assert kwargs["generation_config"]["response_mime_type"] == "application/json"
    assert "temperature" in kwargs["generation_config"]
    
    # Verify Vertex AI initialization was triggered
    mock_init_vertex.assert_called_once()
