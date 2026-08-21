import operator
from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

class ChunkPlan(BaseModel):
    chunk_id: str = Field(description="Unique ID for this chunk, e.g. chunk-001")
    page_range: str = Field(description="Pages or section covered, e.g. 1-3 or Section 2")
    description: str = Field(description="Summary of concepts and content in this section")

class HolisticPlan(BaseModel):
    primary_classes: List[str] = Field(
        description="The overarching domain ontology classes identified in the document (e.g., PolymerSynthesis, LapShearTest, ChemicalCompound)"
    )
    chunks: List[ChunkPlan] = Field(
        description="The chunking strategy dividing the document into logical sections"
    )

class Triple(BaseModel):
    subject: str = Field(description="Subject entity URI or canonical identifier (e.g. PolymerBlend_A1)")
    subject_class: str = Field(description="Ontology class of the subject (e.g. PolymerSynthesis, Material)")
    predicate: str = Field(description="Ontology predicate or relationship (e.g. hasProperty, testedOn, synthesizedFrom)")
    object: str = Field(description="Object entity URI, canonical identifier, or literal value")
    object_class: str = Field(description="Ontology class of the object or datatype (e.g. LapShearTest, xsd:float)")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence score between 0.0 and 1.0")
    unit: Optional[str] = Field(default=None, description="QUDT/OM unit of measurement if applicable (e.g. MegaPA, Cel, Kilogram)")
    value: Optional[float] = Field(default=None, description="Numeric value if this triple represents a quantitative measurement")
    chunk_id: Optional[str] = Field(default=None, description="ID of the document chunk from which this triple was extracted")
    source_file: Optional[str] = Field(default=None, description="Source document GCS URI or filename")

class ExtractionResult(BaseModel):
    triples: List[Triple] = Field(default_factory=list, description="Extracted RDF knowledge triples with full edge metadata")

class GraphState(TypedDict):
    bucket_name: str
    file_name: str
    document_uri: str
    primary_classes: List[str]
    chunks: List[Dict[str, Any]]
    schema_context: Dict[str, Any]
    extracted_triples: Annotated[List[Dict[str, Any]], operator.add]
    errors: Annotated[List[str], operator.add]
