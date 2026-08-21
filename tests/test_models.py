import pytest
from pydantic import ValidationError
from models import Triple, ChunkPlan, HolisticPlan, ExtractionResult, GraphState

def test_triple_valid_all_fields():
    triple = Triple(
        subject="PolymerBlend_X",
        subject_class="PolymerSynthesis",
        predicate="yieldsProduct",
        object="LapShearSample_01",
        object_class="MaterialSample",
        confidence=0.95,
        unit="MegaPA",
        value=45.5,
        chunk_id="chunk-001",
        source_file="gs://bucket/doc.pdf"
    )
    assert triple.subject == "PolymerBlend_X"
    assert triple.subject_class == "PolymerSynthesis"
    assert triple.predicate == "yieldsProduct"
    assert triple.object == "LapShearSample_01"
    assert triple.object_class == "MaterialSample"
    assert triple.confidence == 0.95
    assert triple.unit == "MegaPA"
    assert triple.value == 45.5
    assert triple.chunk_id == "chunk-001"
    assert triple.source_file == "gs://bucket/doc.pdf"

def test_triple_defaults():
    triple = Triple(
        subject="Compound_A",
        subject_class="ChemicalCompound",
        predicate="hasCASNumber",
        object="108-88-3",
        object_class="xsd:string"
    )
    assert triple.confidence == 1.0
    assert triple.unit is None
    assert triple.value is None
    assert triple.chunk_id is None
    assert triple.source_file is None

def test_triple_confidence_bounds():
    with pytest.raises(ValidationError):
        Triple(
            subject="A", subject_class="C", predicate="p", object="B", object_class="C",
            confidence=1.5
        )
    with pytest.raises(ValidationError):
        Triple(
            subject="A", subject_class="C", predicate="p", object="B", object_class="C",
            confidence=-0.1
        )

def test_chunk_plan_and_holistic_plan():
    cp1 = ChunkPlan(chunk_id="c1", page_range="1-2", description="Introduction")
    cp2 = ChunkPlan(chunk_id="c2", page_range="3-5", description="Methods")
    plan = HolisticPlan(
        primary_classes=["PolymerSynthesis", "LapShearTest"],
        chunks=[cp1, cp2]
    )
    assert len(plan.chunks) == 2
    assert "PolymerSynthesis" in plan.primary_classes

def test_extraction_result():
    result = ExtractionResult(triples=[
        Triple(
            subject="S1", subject_class="C1", predicate="p1", object="O1", object_class="C2",
            confidence=0.9
        )
    ])
    assert len(result.triples) == 1
    assert result.triples[0].subject == "S1"
