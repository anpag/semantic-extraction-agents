import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from schema_retriever import DynamicSchemaRetriever, DEFAULT_DOMAIN_SCHEMAS

@pytest.mark.asyncio
async def test_schema_retriever_fallback_to_domain_registry():
    retriever = DynamicSchemaRetriever()
    schema_slice = await retriever.get_schema_slice(["PolymerSynthesis", "LapShearTest"])
    
    assert "PolymerSynthesis" in schema_slice
    assert "LapShearTest" in schema_slice
    assert "hasReactant" in schema_slice["PolymerSynthesis"]["allowed_predicates"]
    assert "hasAdhesionStrength" in schema_slice["LapShearTest"]["allowed_predicates"]

@pytest.mark.asyncio
async def test_schema_retriever_novel_class_dynamic_generation():
    retriever = DynamicSchemaRetriever()
    schema_slice = await retriever.get_schema_slice(["NovelNanoMaterial"])
    
    assert "NovelNanoMaterial" in schema_slice
    assert "hasProperty" in schema_slice["NovelNanoMaterial"]["allowed_predicates"]

@pytest.mark.asyncio
async def test_schema_retriever_from_reasoning_engine():
    retriever = DynamicSchemaRetriever(reasoning_engine_url="http://mock-reasoning-engine:8080")
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "CustomClass": {
            "class_name": "CustomClass",
            "allowed_predicates": ["customPred"],
            "property_types": {"customPred": "xsd:string"}
        }
    }
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        schema_slice = await retriever.fetch_schema_slice_from_reasoning_engine(["CustomClass"])
        assert schema_slice is not None
        assert "CustomClass" in schema_slice

def test_format_schema_for_prompt():
    retriever = DynamicSchemaRetriever()
    formatted = retriever.format_schema_for_prompt(DEFAULT_DOMAIN_SCHEMAS)
    assert "DYNAMIC ONTOLOGY SCHEMA SLICE" in formatted
    assert "PolymerSynthesis" in formatted
    assert "hasReactant" in formatted

def test_schema_retriever_from_bigquery():
    retriever = DynamicSchemaRetriever(project_id="test-project", bq_dataset="test_dataset")
    mock_bq_client = MagicMock()
    mock_query_job = MagicMock()
    mock_query_job.result.return_value = [
        {
            "class_name": "TestBQClass",
            "property_name": "hasTestProp",
            "property_type": "xsd:string",
            "range_class": "TestRange"
        }
    ]
    mock_bq_client.query.return_value = mock_query_job
    
    with patch.object(retriever, "_get_bq_client", return_value=mock_bq_client):
        schema_slice = retriever.fetch_schema_slice_from_bigquery(["TestBQClass"])
        assert schema_slice is not None
        assert "TestBQClass" in schema_slice
        assert "hasTestProp" in schema_slice["TestBQClass"]["allowed_predicates"]
        assert schema_slice["TestBQClass"]["property_types"]["hasTestProp"] == "TestRange"

