import os
import json
import mimetypes
import sys
from google.cloud import bigquery
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# --- Configuration ---
PROJECT_ID = os.environ.get("PROJECT_ID", "semantic-graph-demo")
LOCATION = os.environ.get("LOCATION", "global") # Using global for 3.5-flash availability
DATASET_ID = "kg_ontology_production"

# Enterprise standard: MUST use 3.5-flash or 3.1-pro
MODEL_NAME = "gemini-3.5-flash"

def initialize_vertex():
    """Initializes the Vertex AI SDK."""
    print(f"Initializing Vertex AI in {PROJECT_ID}/{LOCATION} using {MODEL_NAME}...")
    vertexai.init(project=PROJECT_ID, location=LOCATION)

def fetch_ontology_context(bq_client: bigquery.Client) -> tuple[str, bool]:
    """
    Fetches the allowed Nodes and Edges from the BigQuery Production Graph
    to build the Dynamic System Prompt. Returns (context, is_empty).
    """
    print(f"Fetching ontology context from {DATASET_ID}...")
    
    # 1. Fetch allowed Node Classes
    query_nodes = f"SELECT class_name, definition FROM `{PROJECT_ID}.{DATASET_ID}.node_classes`"
    nodes_df = bq_client.query(query_nodes).to_dataframe()
    
    # 2. Fetch allowed Edge Rules
    query_edges = f"SELECT source_uri, relationship_type, target_uri FROM `{PROJECT_ID}.{DATASET_ID}.edge_rules`"
    edges_df = bq_client.query(query_edges).to_dataframe()
    
    if nodes_df.empty and edges_df.empty:
        return "", True
    
    # Build the context string
    context = "ALLOWED NODE CLASSES AND DEFINITIONS:\n"
    for _, row in nodes_df.iterrows():
        context += f"- Class: {row['class_name']} | Definition: {row['definition']}\n"
        
    context += "\nALLOWED RELATIONSHIPS (Source -> Relationship -> Target):\n"
    for _, row in edges_df.iterrows():
        context += f"- {row['source_uri']} -> {row['relationship_type']} -> {row['target_uri']}\n"
        
    return context, False

def build_system_prompt(ontology_context: str) -> str:
    """Builds the strict extraction prompt."""
    return f"""
    You are an expert Data Extraction Agent acting as the "Dirty Extraction Stage" for a Semantic Clean Room.
    Your task is to analyze the provided text and extract raw Entities and Relationships.
    
    CRITICAL RULES:
    1. Extract exactly what is written in the text. Do not attempt to guess or canonicalize standard Units of Measurement or canonical names.
    2. However, the TYPES of entities and relationships you extract must be strictly bounded by the allowed ontology below.
    3. If you find tacit knowledge or insights that don't fit the strict graph topology, place them in `unbound_knowledge`.
    4. EXHAUSTIVE EXTRACTION IS MANDATORY: You MUST extract relationships for ALL entities across ALL columns and ALL rows. Do not stop after processing the first column.
    
    ONTOLOGY CONTEXT:
    {ontology_context}
    
    OUTPUT SCHEMA:
    You MUST output valid JSON conforming exactly to this structure:
    {{
        "extraction_plan": "Step-by-step plan to ensure exhaustive extraction of all nodes and edges across all columns/rows without skipping any",
        "extracted_nodes": [
            {{"entity_name": "exact text from doc", "ontology_class": "must match an allowed class", "raw_properties": {{"raw_unit": "mls", "value": "10"}}}}
        ],
        "extracted_edges": [
            {{"source_entity": "must match a node", "target_entity": "must match a node", "relationship_type": "must match allowed relationship", "evidence": "textual evidence"}}
        ],
        "unbound_knowledge": [
            {{"insight": "critical tacit knowledge", "category": "inferred category"}}
        ]
    }}
    """

def build_open_extraction_prompt() -> str:
    """Builds the prompt for Automagic Ontology Generation when starting from a blank slate."""
    return f"""
    You are an expert Enterprise Knowledge Graph Architect acting as the "Automagic Ontology Generator" for a Semantic Clean Room.
    Your task is to analyze the provided text, infer a robust baseline Ontology, and extract the raw Entities and Relationships.
    
    CRITICAL INSPIRATION RULES:
    1. You MUST draw deep inspiration from the Allotrope Foundation Ontology (AFO) when categorizing materials, properties, and equipment.
    2. You MUST draw deep inspiration from the Chemical Methods Ontology (CHMO) when categorizing assays, test methodologies, and experimental procedures.
    3. EXHAUSTIVE EXTRACTION IS MANDATORY: You MUST extract relationships for ALL entities across ALL columns and ALL rows. Do not stop after processing the first column.
    
    OUTPUT SCHEMA:
    You MUST output valid JSON conforming exactly to this structure:
    {{
        "extraction_plan": "Step-by-step plan to ensure exhaustive extraction of all nodes and edges across all columns/rows without skipping any. MUST explicitly plan how to connect EVERY single node into a unified graph. MUST ensure that EVERY inferred ontology node class participates in at least one edge_rule (no free/isolated ontology classes).",
        "inferred_ontology": {{
            "node_classes": [
                {{"class_name": "e.g. Formulation", "definition": "A mixture of components...", "uri": "afo:Formulation", "synonyms": "mix, blend", "example": "FH-001"}}
            ],
            "edge_rules": [
                {{"domain_class": "Formulation", "range_class": "LapShearTest", "relationship_type": "undergoes_test"}}
            ]
        }},
        "extracted_nodes": [
            {{"entity_name": "exact text from doc", "ontology_class": "must match an inferred class", "raw_properties": {{"raw_unit": "mls", "value": "10"}}}}
        ],
        "extracted_edges": [
            {{"source_entity": "must match a node", "target_entity": "must match a node", "relationship_type": "must match an inferred relationship_type", "evidence": "textual evidence"}}
        ]
    }}
    CRITICAL: 
    1. YOU MUST EXTRACT `extracted_edges` linking the `extracted_nodes` together based on your `edge_rules`. IF `extracted_edges` IS EMPTY, YOU HAVE FAILED. EVERY EXTRACTED NODE MUST BE CONNECTED.
    2. THE INFERRED ONTOLOGY MUST BE FULLY CONNECTED. EVERY single class in `node_classes` MUST be used in at least one `edge_rule`. IF THERE ARE ANY ISOLATED NODE CLASSES IN THE ONTOLOGY, YOU HAVE FAILED.
    3. DO NOT create abstract, top-level hierarchical categories (e.g. 'Equipment', 'Chemical Test', 'Physical Test') unless you explicitly extract nodes belonging to them AND connect them via edge rules. Every node_class must be grounded in actual extracted entities and relationships.
    """

def extract_triples_from_file(file_path: str):
    """Main execution flow for the agent processing a multimodal file."""
    bq_client = bigquery.Client(project=PROJECT_ID)
    initialize_vertex()
    
    # 1. Dynamically build context from BigQuery
    ontology_context, is_empty = fetch_ontology_context(bq_client)
    
    if is_empty:
        print("Ontology is empty! Switching to Automagic Open Extraction (AFO/CHMO inspired)...")
        system_instruction = build_open_extraction_prompt()
    else:
        print("Ontology found. Using Strict Extraction...")
        system_instruction = build_system_prompt(ontology_context)
    
    # 2. Instantiate Gemini
    model = GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=[system_instruction]
    )
    
    # 3. Handle file formats
    if file_path.endswith('.xlsx'):
        import pandas as pd
        # Read all sheets into a markdown representation to preserve spatial structure
        dfs = pd.read_excel(file_path, sheet_name=None)
        content = "The following is a Markdown representation of the Excel file:\n\n"
        for sheet_name, df in dfs.items():
            content += f"### Sheet: {sheet_name}\n\n"
            headers = [str(c) for c in df.columns]
            content += "| " + " | ".join(headers) + " |\n"
            content += "| " + " | ".join(["---"] * len(headers)) + " |\n"
            for _, row in df.iterrows():
                content += "| " + " | ".join([str(x).replace("|", "\\|").replace("\n", " ") if pd.notna(x) else "" for x in row]) + " |\n"
            content += "\n"
        contents = [content]
    else:
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = 'application/octet-stream'
                
        with open(file_path, "rb") as f:
            file_bytes = f.read()
            
        document_part = Part.from_data(data=file_bytes, mime_type=mime_type)
        contents = [document_part]
    
    # 4. Execute Extraction (Enforcing JSON output)
    print(f"Executing extraction via Gemini for {os.path.basename(file_path)}...")
    response = model.generate_content(
        contents,
        generation_config={"response_mime_type": "application/json", "temperature": 0.0}
    )
    
    parsed_response = json.loads(response.text)
    
    # 5. Insert Inferred Ontology (if open extraction)
    if is_empty and "inferred_ontology" in parsed_response:
        print("Saving inferred ontology to BigQuery staging tables...")
        nodes = parsed_response["inferred_ontology"].get("node_classes", [])
        edges = parsed_response["inferred_ontology"].get("edge_rules", [])
        
        if nodes:
            job = bq_client.load_table_from_json(nodes, f"{PROJECT_ID}.kg_ontology_staging.onto_classes")
            job.result()
        if edges:
            job = bq_client.load_table_from_json(edges, f"{PROJECT_ID}.kg_ontology_staging.onto_rules")
            job.result()
            
    # 6. Insert Extracted Data into Staging
    print("Saving extracted triples to raw_extractions_landing...")
    query = f"""
        INSERT INTO `{PROJECT_ID}.kg_graph_staging.raw_extractions_landing`
        (source_file, extracted_nodes, extracted_edges, unbound_knowledge)
        VALUES (
            @source_file,
            PARSE_JSON(@nodes_str),
            PARSE_JSON(@edges_str),
            PARSE_JSON(@unbound_str)
        )
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("source_file", "STRING", os.path.basename(file_path)),
            bigquery.ScalarQueryParameter("nodes_str", "STRING", json.dumps(parsed_response.get("extracted_nodes", []))),
            bigquery.ScalarQueryParameter("edges_str", "STRING", json.dumps(parsed_response.get("extracted_edges", []))),
            bigquery.ScalarQueryParameter("unbound_str", "STRING", json.dumps(parsed_response.get("unbound_knowledge", [])))
        ]
    )
    bq_client.query(query, job_config=job_config).result()
    
    # 7. Insert Unbound Knowledge into DLQ (if strict extraction)
    unbound = parsed_response.get("unbound_knowledge", [])
    if not is_empty and unbound:
        print("Saving unbound knowledge to dlq_semantic_failures...")
        dlq_rows = []
        import datetime
        now_str = datetime.datetime.utcnow().isoformat()
        for ub in unbound:
            dlq_rows.append({
                "file_id": os.path.basename(file_path),
                "page_num": 1,
                "anomaly_type": "Unbound Knowledge",
                "raw_value": ub.get("insight", ""),
                "context": ub.get("category", ""),
                "inferred_value": None,
                "confidence_score": 1.0,
                "flagged_at": now_str
            })
        if dlq_rows:
            job = bq_client.load_table_from_json(dlq_rows, f"{PROJECT_ID}.{DATASET_ID}.dlq_semantic_failures")
            job.result()
    
    return response.text

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Process a single file if passed as argument
        file_path = sys.argv[1]
        print(f"\n--- Processing {os.path.basename(file_path)} ---")
        try:
            result_json = extract_triples_from_file(file_path)
            parsed_result = json.loads(result_json)
            parsed_result["source_file"] = os.path.basename(file_path)
            print("EXTRACTION COMPLETE.")
            # We print the parsed result so the backend can capture it if needed
            print(json.dumps(parsed_result, indent=2))
        except Exception as e:
            print(f"Error during extraction for {file_path}: {e}")
            sys.exit(1)
    else:
        # Default demo behavior
        demo_files = [
            "../Demo/02_Demo Data/Demo Data - Project FlyHigh Gloo.xlsx",
            "../Demo/02_Demo Data/Demo Data - Project HotDump Gloo.xlsx",
            "../Demo/02_Demo Data/Demo Data - Project InstaDust Gloo.pdf"
        ]
        
        all_results = []
        
        for file_path in demo_files:
            print(f"\n--- Processing {os.path.basename(file_path)} ---")
            try:
                result_json = extract_triples_from_file(file_path)
                parsed_result = json.loads(result_json)
                parsed_result["source_file"] = os.path.basename(file_path)
                all_results.append(parsed_result)
                print(json.dumps(parsed_result, indent=2))
            except Exception as e:
                print(f"Error during extraction for {file_path}: {e}")
                
        # Save all results to a JSON file for the pipeline to ingest
        with open("extracted_real_data.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print("\nSaved all extracted results to extracted_real_data.json")
