import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from models import HolisticPlan, ChunkPlan, ExtractionResult, Triple, GraphState
import graph

@pytest.mark.asyncio
async def test_holistic_planner_async():
    mock_plan = HolisticPlan(
        primary_classes=["PolymerSynthesis"],
        chunks=[
            ChunkPlan(chunk_id="chunk-1", page_range="1-2", description="Overview"),
            ChunkPlan(chunk_id="chunk-2", page_range="3-4", description="Synthesis")
        ]
    )
    
    state: GraphState = {
        "bucket_name": "test-bucket",
        "file_name": "test.pdf",
        "document_uri": "gs://test-bucket/test.pdf",
        "primary_classes": [],
        "chunks": [],
        "schema_context": {},
        "extracted_triples": [],
        "errors": []
    }
    
    with patch("graph._invoke_structured_llm_async", new_callable=AsyncMock) as mock_llm_call:
        mock_llm_call.return_value = mock_plan
        result = await graph.holistic_planner_async(state)
        
        assert "PolymerSynthesis" in result["primary_classes"]
        assert len(result["chunks"]) == 2
        assert "schema_context" in result

@pytest.mark.asyncio
async def test_targeted_extraction_async_concurrency():
    state: GraphState = {
        "bucket_name": "test-bucket",
        "file_name": "test.pdf",
        "document_uri": "gs://test-bucket/test.pdf",
        "primary_classes": ["PolymerSynthesis"],
        "chunks": [
            {"chunk_id": "chunk-1", "page_range": "1-2", "description": "Section 1"},
            {"chunk_id": "chunk-2", "page_range": "3-4", "description": "Section 2"},
            {"chunk_id": "chunk-3", "page_range": "5-6", "description": "Section 3"}
        ],
        "schema_context": {},
        "extracted_triples": [],
        "errors": []
    }
    
    mock_extraction_res = ExtractionResult(
        triples=[
            Triple(
                subject="Polymer_A",
                subject_class="PolymerSynthesis",
                predicate="yieldsProduct",
                object="Blend_1",
                object_class="Material",
                confidence=0.98,
                unit="MPa",
                value=42.0
            )
        ]
    )
    
    with patch("graph._invoke_structured_llm_async", new_callable=AsyncMock) as mock_llm_call:
        mock_llm_call.return_value = mock_extraction_res
        res = await graph.targeted_extraction_async(state)
        
        assert len(res["extracted_triples"]) == 3
        assert res["extracted_triples"][0]["subject"] == "Polymer_A"
        assert res["extracted_triples"][0]["chunk_id"] == "chunk-1"
        assert res["extracted_triples"][0]["source_file"] == "gs://test-bucket/test.pdf"

@pytest.mark.asyncio
async def test_targeted_extraction_error_resilience():
    state: GraphState = {
        "bucket_name": "test-bucket",
        "file_name": "test.pdf",
        "document_uri": "gs://test-bucket/test.pdf",
        "primary_classes": ["PolymerSynthesis"],
        "chunks": [
            {"chunk_id": "chunk-1", "page_range": "1-2", "description": "Section 1"}
        ],
        "schema_context": {},
        "extracted_triples": [],
        "errors": []
    }
    
    with patch("graph._invoke_structured_llm_async", new_callable=AsyncMock) as mock_llm_call:
        mock_llm_call.side_effect = RuntimeError("Rate limit exceeded")
        res = await graph.targeted_extraction_async(state)
        
        assert len(res["extracted_triples"]) == 0
        assert len(res["errors"]) > 0
        assert "Rate limit exceeded" in res["errors"][0]
