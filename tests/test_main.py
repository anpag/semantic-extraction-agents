import json
import base64
import pytest
from unittest.mock import patch, MagicMock
import main

@pytest.fixture
def client():
    main.app.config['TESTING'] = True
    with main.app.test_client() as client:
        yield client

def test_health_check(client):
    response = client.get('/healthz')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "healthy"
    assert data["service"] == "extraction-agents"

def test_direct_extract_missing_param(client):
    response = client.post('/extract', json={})
    assert response.status_code == 400

@patch('main.process_document_with_graph')
def test_direct_extract_success(mock_process, client):
    mock_process.return_value = {
        "document_uri": "gs://bucket/file.pdf",
        "primary_classes": ["PolymerSynthesis"],
        "chunks": [{"chunk_id": "c1"}],
        "extracted_triples": [{"subject": "A", "predicate": "p", "object": "B"}],
        "errors": []
    }
    
    response = client.post('/extract', json={"bucket_name": "bucket", "file_name": "file.pdf"})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "success"
    assert len(data["extracted_triples"]) == 1

def test_pubsub_push_invalid_format(client):
    response = client.post('/', json={})
    assert response.status_code == 400
    
    response = client.post('/', json={"wrong_key": "val"})
    assert response.status_code == 400

@patch('main.process_document_with_graph')
@patch('main.get_publisher')
def test_pubsub_push_success(mock_get_publisher, mock_process, client):
    mock_process.return_value = {
        "document_uri": "gs://bucket/file.pdf",
        "primary_classes": ["PolymerSynthesis"],
        "extracted_triples": [],
        "errors": []
    }
    mock_pub = MagicMock()
    mock_topic = "projects/test/topics/raw-graph-events"
    mock_future = MagicMock()
    mock_future.result.return_value = "msg-12345"
    mock_pub.publish.return_value = mock_future
    mock_get_publisher.return_value = (mock_pub, mock_topic)
    
    payload = {"bucket": "bucket", "name": "file.pdf"}
    encoded_data = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")
    envelope = {"message": {"data": encoded_data}}
    
    response = client.post('/', json=envelope)
    assert response.status_code == 204
    mock_process.assert_called_once_with("bucket", "file.pdf")
