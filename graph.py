import os
import logging
import asyncio
from typing import Dict, Any, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_google_vertexai import ChatVertexAI

from models import ChunkPlan, HolisticPlan, Triple, ExtractionResult, GraphState
from schema_retriever import DynamicSchemaRetriever
from document_loader import build_multimodal_content

logger = logging.getLogger(__name__)

# Configurable environment settings - zero hardcoded IDs
PROJECT_ID = os.environ.get("PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
LOCATION = os.environ.get("LOCATION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-1.5-flash")
MAX_CONCURRENT_CHUNKS = int(os.environ.get("MAX_CONCURRENT_CHUNKS", "5"))

def get_llm(structured_type=None):
    """Instantiates ChatVertexAI with configurable model, project, and location."""
    kwargs = {
        "model": MODEL_NAME,
        "location": LOCATION,
        "temperature": 0.0,
    }
    if PROJECT_ID:
        kwargs["project"] = PROJECT_ID
    llm = ChatVertexAI(**kwargs)
    if structured_type:
        return llm.with_structured_output(structured_type)
    return llm

# Exponential backoff retry handler for Vertex AI rate limits & transient errors
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=1.0, max=30.0, exp_base=2),
    reraise=True
)
async def _invoke_structured_llm_async(structured_llm, messages: List[Any]):
    """Invokes structured LLM with retry logic."""
    if hasattr(structured_llm, "ainvoke"):
        return await structured_llm.ainvoke(messages)
    return await asyncio.to_thread(structured_llm.invoke, messages)

async def holistic_planner_async(state: GraphState) -> Dict[str, Any]:
    """Agent 1: Ingests document / PDF, establishes overarching domain classes and chunking strategy."""
    doc_uri = state.get("document_uri", "")
    logger.info(f"Running Holistic Planner on {doc_uri}...")
    
    structured_llm = get_llm(HolisticPlan)
    
    prompt = (
        f"Analyze the document at {doc_uri}.\n"
        "Identify the primary domain ontology classes (e.g., PolymerSynthesis, LapShearTest, Material, ChemicalCompound).\n"
        "Devise a chunking plan dividing the document into logical sections (e.g., Introduction, Synthesis Method, Characterization, Results)."
    )
    
    contents = build_multimodal_content(doc_uri, prompt)
    messages = [HumanMessage(content=contents)]
    
    try:
        result: HolisticPlan = await _invoke_structured_llm_async(structured_llm, messages)
        chunks_dict = [
            {"chunk_id": c.chunk_id, "page_range": c.page_range, "description": c.description}
            for c in result.chunks
        ]
        
        # Initialize schema retriever and fetch schema slice for detected primary classes
        retriever = DynamicSchemaRetriever(project_id=PROJECT_ID)
        schema_slice = await retriever.get_schema_slice(result.primary_classes)
        
        return {
            "primary_classes": result.primary_classes,
            "chunks": chunks_dict,
            "schema_context": schema_slice
        }
    except Exception as e:
        logger.error(f"Holistic Planner failed for {doc_uri}: {e}")
        return {
            "primary_classes": ["DocumentSection"],
            "chunks": [{"chunk_id": "chunk-001", "page_range": "all", "description": "Full document content"}],
            "schema_context": {},
            "errors": [f"Holistic Planner error: {str(e)}"]
        }

def holistic_planner(state: GraphState) -> Dict[str, Any]:
    """Synchronous wrapper for holistic planner node."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(holistic_planner_async(state))
        else:
            return asyncio.run(holistic_planner_async(state))
    except Exception:
        return asyncio.run(holistic_planner_async(state))

async def _extract_single_chunk(
    chunk: Dict[str, Any],
    state: GraphState,
    schema_prompt: str,
    semaphore: asyncio.Semaphore,
    structured_llm
) -> Dict[str, Any]:
    """Extracts triples for a single chunk concurrently under semaphore and rate-limit retry."""
    async with semaphore:
        chunk_id = chunk.get("chunk_id", "unknown")
        page_range = chunk.get("page_range", "")
        description = chunk.get("description", "")
        doc_uri = state.get("document_uri", "")
        
        logger.info(f"Targeted extraction on chunk {chunk_id} ({page_range}): {description}")
        
        prompt = f"""
        Extract knowledge triples from document section: {description} (Pages/Section: {page_range}).
        Source Document: {doc_uri}
        
        CRITICAL RULES:
        1. Extract complete, verified knowledge triples with strict typing:
           - subject: URI or canonical identifier
           - subject_class: identified ontology class
           - predicate: valid ontology relationship (e.g. hasProperty, testedOn, yieldsProduct)
           - object: URI, identifier, or literal
           - object_class: target class or datatype (e.g. LapShearTest, xsd:float, ChemicalCompound)
           - confidence: extraction certainty score (0.0 - 1.0)
           - unit: QUDT/OM unit string if numeric measurement (e.g., MegaPA, Cel, MPa, g/mol)
           - value: numeric value if numeric measurement
           - chunk_id: '{chunk_id}'
           - source_file: '{doc_uri}'
        
        2. Strictly conform to this Dynamic Ontology Schema:
        {schema_prompt}
        """
        
        contents = build_multimodal_content(doc_uri, prompt)
        messages = [HumanMessage(content=contents)]
        
        try:
            result: ExtractionResult = await _invoke_structured_llm_async(structured_llm, messages)
            triples = []
            for t in result.triples:
                t_dict = t.model_dump()
                t_dict["chunk_id"] = chunk_id
                t_dict["source_file"] = doc_uri
                triples.append(t_dict)
            return {"triples": triples, "errors": []}
        except Exception as e:
            err_msg = f"Failed to extract chunk {chunk_id}: {str(e)}"
            logger.error(err_msg)
            return {"triples": [], "errors": [err_msg]}

async def targeted_extraction_async(state: GraphState) -> Dict[str, Any]:
    """Agent 1.5 & Agent 2: Concurrently processes document chunks with dynamic schema and exponential backoff."""
    chunks = state.get("chunks", [])
    if not chunks:
        return {"extracted_triples": [], "errors": ["No chunks available for extraction."]}
    
    logger.info(f"Running Targeted Extraction on {len(chunks)} chunks concurrently (max concurrency: {MAX_CONCURRENT_CHUNKS})...")
    
    # Retrieve dynamic schema
    retriever = DynamicSchemaRetriever(project_id=PROJECT_ID)
    schema_context = state.get("schema_context")
    if not schema_context:
        schema_context = await retriever.get_schema_slice(state.get("primary_classes", []))
    schema_prompt = retriever.format_schema_for_prompt(schema_context)
    
    structured_llm = get_llm(ExtractionResult)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHUNKS)
    
    tasks = [
        _extract_single_chunk(chunk, state, schema_prompt, semaphore, structured_llm)
        for chunk in chunks
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    all_triples = []
    all_errors = []
    
    for r in results:
        if isinstance(r, Exception):
            all_errors.append(str(r))
        elif isinstance(r, dict):
            all_triples.extend(r.get("triples", []))
            all_errors.extend(r.get("errors", []))
            
    return {
        "extracted_triples": all_triples,
        "errors": all_errors
    }

def targeted_extraction(state: GraphState) -> Dict[str, Any]:
    """Synchronous wrapper for targeted extraction node."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(targeted_extraction_async(state))
        else:
            return asyncio.run(targeted_extraction_async(state))
    except Exception:
        return asyncio.run(targeted_extraction_async(state))

# --- Graph Compilation ---
workflow = StateGraph(GraphState)
workflow.add_node("planner", holistic_planner)
workflow.add_node("extractor", targeted_extraction)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "extractor")
workflow.add_edge("extractor", END)

app = workflow.compile()

async def aprocess_document_with_graph(bucket_name: str, file_name: str) -> Dict[str, Any]:
    """Asynchronous entrypoint for the LangGraph workflow."""
    doc_uri = f"gs://{bucket_name}/{file_name}" if bucket_name else file_name
    initial_state = {
        "bucket_name": bucket_name or "",
        "file_name": file_name,
        "document_uri": doc_uri,
        "primary_classes": [],
        "chunks": [],
        "schema_context": {},
        "extracted_triples": [],
        "errors": []
    }
    
    logger.info(f"Invoking LangGraph Workflow for {doc_uri}...")
    final_state = await app.ainvoke(initial_state)
    return final_state

def process_document_with_graph(bucket_name: str, file_name: str) -> Dict[str, Any]:
    """Synchronous entrypoint for the LangGraph workflow."""
    doc_uri = f"gs://{bucket_name}/{file_name}" if bucket_name else file_name
    initial_state = {
        "bucket_name": bucket_name or "",
        "file_name": file_name,
        "document_uri": doc_uri,
        "primary_classes": [],
        "chunks": [],
        "schema_context": {},
        "extracted_triples": [],
        "errors": []
    }
    
    logger.info(f"Invoking LangGraph Workflow for {doc_uri}...")
    final_state = app.invoke(initial_state)
    return final_state
