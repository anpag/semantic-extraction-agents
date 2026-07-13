import os
import json
from google.cloud import bigquery

PROJECT_ID = os.environ.get("PROJECT_ID", "semantic-graph-demo")
DATASET_ID = "kg_graph_staging"
TABLE_ID = "raw_extractions_landing"

def ingest_to_bq(json_file_path):
    print(f"Starting ingestion to {PROJECT_ID}.{DATASET_ID}.{TABLE_ID}...")
    client = bigquery.Client(project=PROJECT_ID)
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

    # Define schema with JSON columns
    schema = [
        bigquery.SchemaField("source_file", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("extracted_nodes", "JSON", mode="NULLABLE"),
        bigquery.SchemaField("extracted_edges", "JSON", mode="NULLABLE"),
        bigquery.SchemaField("unbound_knowledge", "JSON", mode="NULLABLE"),
    ]

    # Create table if it doesn't exist
    table = bigquery.Table(table_ref, schema=schema)
    try:
        client.get_table(table_ref)
        print("Table exists, appending data.")
    except Exception:
        print("Table not found, creating it...")
        client.create_table(table)

    # Read JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Format data for BigQuery insertion (JSON columns expect strings or dicts depending on library version, 
    # but the BQ JSON type accepts python dicts during insert_rows_json or load_table_from_json)
    
    # We will use load_table_from_json for robust batch loading
    formatted_data = []
    for row in data:
        formatted_data.append({
            "source_file": row.get("source_file", "unknown"),
            # We dump to string because sometimes the load job prefers stringified JSON for JSON columns,
            # but the bq library handles dicts for JSON columns well if schema is explicitly passed.
            "extracted_nodes": row.get("extracted_nodes", []),
            "extracted_edges": row.get("extracted_edges", []),
            "unbound_knowledge": row.get("unbound_knowledge", [])
        })

    job_config = bigquery.LoadJobConfig(
        schema=schema,
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND
    )

    # load_table_from_json handles the dict to NDJSON conversion in memory
    job = client.load_table_from_json(formatted_data, table_ref, job_config=job_config)
    job.result()  # Wait for the job to complete
    
    print(f"Loaded {job.output_rows} rows into {table_ref}.")

if __name__ == "__main__":
    file_path = "extracted_real_data.json"
    if os.path.exists(file_path):
        ingest_to_bq(file_path)
    else:
        print(f"File not found: {file_path}")
