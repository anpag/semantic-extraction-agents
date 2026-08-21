import os
import json
import base64
import logging
from flask import Flask, request, jsonify
from google.cloud import pubsub_v1
from graph import process_document_with_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

PROJECT_ID = os.environ.get("PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
OUTPUT_TOPIC = os.environ.get("OUTPUT_TOPIC", "raw-graph-events")

_publisher = None
_topic_path = None

def get_publisher():
    global _publisher, _topic_path
    if _publisher is None and PROJECT_ID and OUTPUT_TOPIC:
        try:
            _publisher = pubsub_v1.PublisherClient()
            _topic_path = _publisher.topic_path(PROJECT_ID, OUTPUT_TOPIC)
        except Exception as e:
            logger.warning(f"Could not initialize Pub/Sub publisher: {e}")
    return _publisher, _topic_path

@app.route("/healthz", methods=["GET"])
def health_check():
    """Liveness and readiness probe endpoint."""
    return jsonify({"status": "healthy", "service": "extraction-agents"}), 200

@app.route("/extract", methods=["POST"])
def direct_extract():
    """Direct REST extraction endpoint for testing and synchronous orchestration."""
    body = request.get_json(silent=True) or {}
    bucket_name = body.get("bucket_name", "")
    file_name = body.get("file_name") or body.get("document_uri")
    
    if not file_name:
        return jsonify({"error": "Missing 'file_name' or 'document_uri' parameter"}), 400
        
    try:
        final_state = process_document_with_graph(bucket_name, file_name)
        return jsonify({
            "status": "success",
            "source_file": final_state.get("document_uri"),
            "primary_classes": final_state.get("primary_classes", []),
            "chunks_processed": len(final_state.get("chunks", [])),
            "extracted_triples": final_state.get("extracted_triples", []),
            "errors": final_state.get("errors", [])
        }), 200
    except Exception as e:
        logger.error(f"Direct extraction failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["POST"])
def pubsub_push():
    """Receives Pub/Sub push events from Eventarc when files are uploaded to GCS."""
    envelope = request.get_json(silent=True)
    if not envelope:
        msg = "no Pub/Sub message received"
        logger.error(f"error: {msg}")
        return f"Bad Request: {msg}", 400

    if not isinstance(envelope, dict) or "message" not in envelope:
        msg = "invalid Pub/Sub message format"
        logger.error(f"error: {msg}")
        return f"Bad Request: {msg}", 400

    pubsub_message = envelope["message"]
    
    if isinstance(pubsub_message, dict) and "data" in pubsub_message:
        try:
            data = base64.b64decode(pubsub_message["data"]).decode("utf-8")
            data_json = json.loads(data)
            
            # Eventarc Cloud Storage event payload
            bucket_name = data_json.get("bucket", "")
            file_name = data_json.get("name", "")
            
            if not file_name:
                logger.warning("Pub/Sub message data missing 'name'. Ignored.")
                return ("", 204)
                
            logger.info(f"Triggered extraction for gs://{bucket_name}/{file_name}")
            
            # Execute the LangGraph workflow
            final_state = process_document_with_graph(bucket_name, file_name)
            
            # Publish the aggregated triples to Pub/Sub
            payload = {
                "source_file": final_state.get("document_uri", f"gs://{bucket_name}/{file_name}"),
                "primary_classes": final_state.get("primary_classes", []),
                "extracted_triples": final_state.get("extracted_triples", []),
                "errors": final_state.get("errors", [])
            }
            
            pub, topic = get_publisher()
            if pub and topic:
                message_bytes = json.dumps(payload).encode("utf-8")
                future = pub.publish(topic, data=message_bytes)
                message_id = future.result()
                logger.info(f"Successfully published extraction event to Pub/Sub. Message ID: {message_id}")
            else:
                logger.info(f"Extraction completed with {len(payload['extracted_triples'])} triples (Pub/Sub publisher not configured)")
            
        except Exception as e:
            logger.error(f"Error processing document: {e}", exc_info=True)
            # Returning 500 will signal Pub/Sub to retry the message
            return f"Internal Server Error: {e}", 500
            
    return ("", 204)

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=PORT, debug=False)
