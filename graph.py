import os
import json
import logging
from typing import TypedDict, List, Dict, Any, Annotated
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_vertexai import ChatVertexAI
from pydantic import BaseModel, Field

PROJECT_ID = os.environ.get("PROJECT_ID", "identity-res-e2e-10022026")
LOCATION = os.environ.get("LOCATION", "global")
MODEL_NAME = "gemini-3.5-flash"

# --- State Definition ---
class GraphState(TypedDict):
    bucket_name: str
    file_name: str
    document_uri: str
    primary_classes: List[str]
    chunks: List[Dict[str, Any]]
    extracted_nodes: Annotated[list, operator.add]
    extracted_edges: Annotated[list, operator.add]
    unbound_knowledge: Annotated[list, operator.add]
    errors: Annotated[list, operator.add]

# --- Structured Outputs ---
class ChunkPlan(BaseModel):
    chunk_id: str = Field(description="Unique ID for this chunk")
    page_range: str = Field(description="Pages covered, e.g. 1-5")
    description: str = Field(description="What this section is about")

class HolisticPlan(BaseModel):
    primary_classes: List[str] = Field(description="The overarching domain classes for this document")
    chunks: List[ChunkPlan] = Field(description="The chunking strategy for the document")

class ExtractedNode(BaseModel):
    entity_name: str = Field(description="The unique canonical name or label of the entity")
    ontology_class: str = Field(description="The exact ontology class name (e.g., PolymerSynthesis, LapShearTest)")
    raw_properties: Dict[str, Any] = Field(default_factory=dict, description="Key-value property pairs extracted for this entity")

class ExtractedEdge(BaseModel):
    source_entity: str = Field(description="The source entity name")
    target_entity: str = Field(description="The target entity name")
    relationship_type: str = Field(description="The ontological relationship predicate (e.g., hasProperty, testedOn)")
    evidence: str = Field(description="Textual evidence or quote from the document supporting this edge")

class UnboundInsight(BaseModel):
    insight: str = Field(description="Abstract summary or high-level finding that does not map directly to an ontology triple")
    category: str = Field(description="Inferred category or domain of this insight")

class ExtractionResult(BaseModel):
    extracted_nodes: List[ExtractedNode] = Field(default_factory=list, description="Extracted entity nodes")
    extracted_edges: List[ExtractedEdge] = Field(default_factory=list, description="Extracted relationship edges")
    unbound_knowledge: List[UnboundInsight] = Field(default_factory=list, description="Unstructured insights")

# --- Nodes ---
def holistic_planner(state: GraphState) -> Dict[str, Any]:
    """Agent 1: Reads the document and establishes the overarching classes and chunking strategy."""
    logging.info(f"Running Holistic Planner on {state['document_uri']}...")
    
    llm = ChatVertexAI(model=MODEL_NAME, project=PROJECT_ID, location=LOCATION, temperature=0.0)
    structured_llm = llm.with_structured_output(HolisticPlan)
    
    # In a real scenario, we'd pass the actual PDF bytes or URI to Gemini.
    # We simulate passing the document URI for context.
    prompt = f"Analyze the document at {state['document_uri']}. Identify the overarching ontology classes (e.g., PolymerSynthesis, LapShearTest) and create a chunking strategy based on logical sections (Introduction, Methods, Results)."
    
    result = structured_llm.invoke([HumanMessage(content=prompt)])
    
    chunks_dict = [{"chunk_id": c.chunk_id, "page_range": c.page_range, "description": c.description} for c in result.chunks]
    
    return {
        "primary_classes": result.primary_classes,
        "chunks": chunks_dict
    }

def targeted_extraction(state: GraphState) -> Dict[str, Any]:
    """Agent 1.5 & Agent 2: For each chunk, retrieves local schema and extracts typed nodes/edges."""
    logging.info(f"Running Targeted Extraction on {len(state['chunks'])} chunks...")
    
    llm = ChatVertexAI(model=MODEL_NAME, project=PROJECT_ID, location=LOCATION, temperature=0.0)
    structured_llm = llm.with_structured_output(ExtractionResult)
    
    all_nodes = []
    all_edges = []
    all_unbound = []
    
    # Map-Reduce parallel loop (simplified for synchronous execution in this mock)
    for chunk in state['chunks']:
        logging.info(f"Processing chunk {chunk['chunk_id']} ({chunk['page_range']}): {chunk['description']}")
        
        # --- AGENT 1.5: Local Schema Retrieval (Mocked) ---
        # Here we would query the API/BigQuery using the primary_classes + chunk.description
        local_schema = f"Schema for {chunk['description']}: Allowed relationships are hasProperty, hasValue, testedOn."
        
        # --- AGENT 2: Targeted Extraction ---
        prompt = f"""
        Extract structured nodes, relationships, and unbound insights from the document section: {chunk['description']} (Pages {chunk['page_range']}).
        Document URI: {state['document_uri']}
        
        CRITICAL RULES:
        1. Extract typed entities with their canonical entity_name, valid ontology_class, and key-value properties.
        2. Extract valid ontological relationships between entities (source_entity, target_entity, relationship_type, evidence).
        3. Adhere to this local schema: {local_schema}
        4. Global document classes identified: {', '.join(state.get('primary_classes', []))}
        """
        
        try:
            result = structured_llm.invoke([HumanMessage(content=prompt)])
            if hasattr(result, "extracted_nodes"):
                all_nodes.extend([n.model_dump() for n in result.extracted_nodes])
            if hasattr(result, "extracted_edges"):
                all_edges.extend([e.model_dump() for e in result.extracted_edges])
            if hasattr(result, "unbound_knowledge"):
                all_unbound.extend([u.model_dump() for u in result.unbound_knowledge])
        except Exception as e:
            logging.error(f"Failed to extract chunk {chunk['chunk_id']}: {e}")
            return {"errors": [str(e)]}
            
    return {
        "extracted_nodes": all_nodes,
        "extracted_edges": all_edges,
        "unbound_knowledge": all_unbound
    }

# --- Graph Compilation ---
workflow = StateGraph(GraphState)

workflow.add_node("planner", holistic_planner)
workflow.add_node("extractor", targeted_extraction)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "extractor")
workflow.add_edge("extractor", END)

app = workflow.compile()

def process_document_with_graph(bucket_name: str, file_name: str) -> Dict[str, Any]:
    """Entrypoint function to run the LangGraph workflow."""
    initial_state = {
        "bucket_name": bucket_name,
        "file_name": file_name,
        "document_uri": f"gs://{bucket_name}/{file_name}",
        "primary_classes": [],
        "chunks": [],
        "extracted_nodes": [],
        "extracted_edges": [],
        "unbound_knowledge": [],
        "errors": []
    }
    
    logging.info("Invoking LangGraph Workflow...")
    final_state = app.invoke(initial_state)
    return final_state
