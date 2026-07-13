import os
import sys
import json
import mimetypes
import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# --- Configuration ---
PROJECT_ID = os.environ.get("PROJECT_ID", "semantic-graph-demo")
LOCATION = os.environ.get("LOCATION", "global")
MODEL_NAME = "gemini-3.5-flash"

def initialize_vertex():
    vertexai.init(project=PROJECT_ID, location=LOCATION)

def build_ontology_draft_prompt(standards):
    standards_list = [s.strip() for s in standards.split(",") if s.strip()]
    standards_description = ""
    
    if "AFO" in standards_list:
        standards_description += "- Allotrope Foundation Ontology (AFO): Guide class naming and URIs for chemicals, formulations, mixing, process parameters, properties, and equipment (e.g. prefix 'afo:').\n"
    if "CHMO" in standards_list:
        standards_description += "- Chemical Methods Ontology (CHMO): Guide naming for assays, analytical methods, measurements, sample preparations, and testing instruments (e.g. prefix 'chmo:').\n"
    if "QUDT" in standards_list:
        standards_description += "- QUDT Quantities, Units, Dimensions and Types: Align physical metrology concepts, standard units of measure, and value mapping.\n"
        
    return f"""
    You are an expert Enterprise Semantic Architect acting as the "Guided Ontology Builder" for Enterprise Adhesive Technologies.
    Your objective is to analyze the uploaded experimental/lab reference file, extract its implicit structural metadata, and design a robust, clean, and extensible baseline Ontology.
    
    CRITICAL ONTOLOGICAL ASSETS TO INCORPORATE:
    {standards_description}
    
    OUTPUT SPECIFICATIONS:
    You must output a single valid JSON object containing:
    1. "node_classes": An array of discovered classes matching the BigQuery staging schema:
       - "class_name": Human-readable class name (e.g. "Formulation", "Experiment", "ViscosityTest"). Use CamelCase.
       - "uri": Standardized URI using appropriate standard prefixes (e.g., "afo:Formulation", "chmo:LapShearTest").
       - "definition": Clear, enterprise-ready definition of what this class represents.
       - "synonyms": Comma-separated list of synonyms found in the lab context (e.g. "mix, blend, recipe").
       - "example": Short string showing an actual sample value (e.g. "FH-001", "EXP-101").
       
    2. "edge_rules": An array of relationship definitions matching the BigQuery staging schema:
       - "domain_class": The source class name (e.g., "Formulation").
       - "range_class": The target class name (e.g., "Ingredient" or "Experiment").
       - "relationship_type": Described in snake_case (e.g., "has_ingredient", "tested_in").
       
    3. "turtle_content": A clean, fully formed W3C RDF Turtle (.ttl) string representing the entire schema, including prefixes (e.g. @prefix rdfs, @prefix skos, @prefix owl, @prefix afo, etc.). All node_classes and edge_rules must be fully serialized in this RDF graph.
    
    CRITICAL SYSTEM INTEGRITY CONSTRAINTS:
    - Ensure EVERY class in "node_classes" participates in at least one rule in "edge_rules" to prevent isolated nodes.
    - All ontology classes must be derived from columns, metrics, or logical groups found inside the reference file.
    - Keep the schema modular, extensible, and professional.
    
    JSON Schema output structure:
    {{
        "node_classes": [
            {{"class_name": "Class", "uri": "prefix:Class", "definition": "Def", "synonyms": "syns", "example": "ex"}}
        ],
        "edge_rules": [
            {{"domain_class": "ClassA", "range_class": "ClassB", "relationship_type": "rel_type"}}
        ],
        "turtle_content": "@prefix ...\\n..."
    }}
    """

def generate_draft(file_path, standards):
    initialize_vertex()
    
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        dfs = pd.read_excel(file_path, sheet_name=None)
        content = "Here is the layout and content of the reference spreadsheet to model as a Markdown representation:\n\n"
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
        
    # 2. Build instruction prompt
    system_instruction = build_ontology_draft_prompt(standards)
    
    # 3. Instantiate model
    model = GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=[system_instruction]
    )
    
    # 4. Generate content
    print(f"Extracting draft ontology from {os.path.basename(file_path)} utilizing standards [{standards}]...")
    response = model.generate_content(
        contents,
        generation_config={"response_mime_type": "application/json", "temperature": 0.1}
    )
    
    return response.text

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_draft_ontology.py <file_path> <comma_separated_standards>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    standards = sys.argv[2]
    
    try:
        draft_json = generate_draft(file_path, standards)
        print("ONTOLOGY_DRAFT_START")
        print(draft_json)
        print("ONTOLOGY_DRAFT_END")
    except Exception as e:
        print(f"Error generating draft ontology: {e}", file=sys.stderr)
        sys.exit(1)
