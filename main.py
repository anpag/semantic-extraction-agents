import os
import json
import base64
import logging
from flask import Flask, request, jsonify
from google.cloud import pubsub_v1
from graph import process_document_with_graph

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

PROJECT_ID = os.environ.get("PROJECT_ID", "identity-res-e2e-10022026")
OUTPUT_TOPIC = os.environ.get("OUTPUT_TOPIC", "raw-graph-events")

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, OUTPUT_TOPIC)

@app.route("/", methods=["POST"])
def pubsub_push():
    """Receives Pub/Sub push events from Eventarc when files are uploaded to GCS."""
    envelope = request.get_json()
    if not envelope:
        msg = "no Pub/Sub message received"
        logging.error(f"error: {msg}")
        return f"Bad Request: {msg}", 400

    if not isinstance(envelope, dict) or "message" not in envelope:
        msg = "invalid Pub/Sub message format"
        logging.error(f"error: {msg}")
        return f"Bad Request: {msg}", 400

    pubsub_message = envelope["message"]
    
    if isinstance(pubsub_message, dict) and "data" in pubsub_message:
        try:
            data = base64.b64decode(pubsub_message["data"]).decode("utf-8")
            data_json = json.loads(data)
            
            # Eventarc Cloud Storage event payload
            bucket_name = data_json.get("bucket")
            file_name = data_json.get("name")
            
            if not bucket_name or not file_name:
                logging.warning("Pub/Sub message data missing 'bucket' or 'name'. Ignored.")
                return ("", 204)
                
            logging.info(f"Triggered extraction for gs://{bucket_name}/{file_name}")
            
            # Execute the LangGraph workflow
            final_state = process_document_with_graph(bucket_name, file_name)
            
            # Publish the aggregated triples to Pub/Sub
            payload = {
                "source_file": f"gs://{bucket_name}/{file_name}",
                "extracted_triples": final_state.get("extracted_triples", []),
                "errors": final_state.get("errors", [])
            }
            
            message_bytes = json.dumps(payload).encode("utf-8")
            future = publisher.publish(topic_path, data=message_bytes)
            message_id = future.result()
            logging.info(f"Successfully published extraction event to Pub/Sub. Message ID: {message_id}")
            
        except Exception as e:
            logging.error(f"Error processing document: {e}")
            # Returning 500 will make Pub/Sub retry the delivery
            return f"Internal Server Error: {e}", 500
            
    return ("", 204)

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=PORT, debug=True)
