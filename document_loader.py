import os
import base64
import logging
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

def build_multimodal_content(document_uri: str, prompt_text: str) -> List[Dict[str, Any]]:
    """
    Constructs the LangChain multimodal input payload.
    Supports GCS URIs (gs://...), local PDF/text files, and raw text.
    """
    if document_uri.startswith("gs://"):
        # LangChain / ChatVertexAI multimodal GCS Part format
        return [
            {
                "type": "media",
                "file_uri": document_uri,
                "mime_type": "application/pdf" if document_uri.lower().endswith(".pdf") else "text/plain"
            },
            {
                "type": "text",
                "text": prompt_text
            }
        ]
    elif os.path.exists(document_uri):
        if document_uri.lower().endswith(".pdf"):
            try:
                with open(document_uri, "rb") as f:
                    encoded_pdf = base64.b64encode(f.read()).decode("utf-8")
                return [
                    {
                        "type": "media",
                        "data": encoded_pdf,
                        "mime_type": "application/pdf"
                    },
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            except Exception as e:
                logger.warning(f"Error encoding local PDF: {e}")
        
        try:
            with open(document_uri, "r", encoding="utf-8", errors="ignore") as f:
                file_text = f.read()
            return [
                {
                    "type": "text",
                    "text": f"{prompt_text}\n\nDocument Content:\n{file_text}"
                }
            ]
        except Exception as e:
            logger.error(f"Error reading local document {document_uri}: {e}")
            return [{"type": "text", "text": prompt_text}]
    else:
        return [
            {
                "type": "text",
                "text": f"{prompt_text}\n\nDocument Text:\n{document_uri}"
            }
        ]
