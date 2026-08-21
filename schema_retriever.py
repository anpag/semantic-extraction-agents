import os
import logging
from typing import List, Dict, Any, Optional
import httpx

try:
    from google.cloud import bigquery
except ImportError:
    bigquery = None

logger = logging.getLogger(__name__)

# Standard domain ontology reference fallback
DEFAULT_DOMAIN_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "PolymerSynthesis": {
        "class_name": "PolymerSynthesis",
        "description": "Chemical process and formulation of polymer materials",
        "allowed_predicates": ["hasReactant", "hasCatalyst", "hasTemperature", "hasSolvent", "yieldsProduct", "hasFormulation"],
        "property_types": {
            "hasReactant": "ChemicalCompound",
            "hasCatalyst": "ChemicalCompound",
            "hasTemperature": "QuantitativeValue",
            "hasSolvent": "ChemicalCompound",
            "yieldsProduct": "PolymerBlend",
            "hasFormulation": "Formulation"
        }
    },
    "LapShearTest": {
        "class_name": "LapShearTest",
        "description": "Mechanical shear strength characterization test",
        "allowed_predicates": ["testedOn", "hasAdhesionStrength", "hasFailureMode", "testedAtTemperature", "hasDisplacementRate"],
        "property_types": {
            "testedOn": "Substrate",
            "hasAdhesionStrength": "QuantitativeValue",
            "hasFailureMode": "FailureMode",
            "testedAtTemperature": "QuantitativeValue",
            "hasDisplacementRate": "QuantitativeValue"
        }
    },
    "ChemicalCompound": {
        "class_name": "ChemicalCompound",
        "description": "Distinct chemical species, monomer, or reagent",
        "allowed_predicates": ["hasMolecularWeight", "hasCASNumber", "hasPurity", "hasSmiles"],
        "property_types": {
            "hasMolecularWeight": "QuantitativeValue",
            "hasCASNumber": "xsd:string",
            "hasPurity": "QuantitativeValue",
            "hasSmiles": "xsd:string"
        }
    },
    "Material": {
        "class_name": "Material",
        "description": "Generic material substance or formulation",
        "allowed_predicates": ["hasProperty", "hasConstituent", "hasDensity", "hasGlassTransitionTemp"],
        "property_types": {
            "hasProperty": "MaterialProperty",
            "hasConstituent": "ChemicalCompound",
            "hasDensity": "QuantitativeValue",
            "hasGlassTransitionTemp": "QuantitativeValue"
        }
    }
}

class DynamicSchemaRetriever:
    """Retrieves and formats ontology schema slices dynamically from BigQuery, Reasoning Engine, or domain registry."""
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        reasoning_engine_url: Optional[str] = None,
        bq_dataset: Optional[str] = None,
        bq_table: str = "ontology_schema"
    ):
        self.project_id = project_id or os.environ.get("PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.reasoning_engine_url = reasoning_engine_url or os.environ.get("REASONING_ENGINE_URL")
        self.bq_dataset = bq_dataset or os.environ.get("BQ_DATASET", "knowledge_graph")
        self.bq_table = bq_table
        self._bq_client = None

    def _get_bq_client(self):
        if self._bq_client is None and bigquery is not None and self.project_id:
            try:
                self._bq_client = bigquery.Client(project=self.project_id)
            except Exception as e:
                logger.warning(f"Could not initialize BigQuery client for schema retrieval: {e}")
        return self._bq_client

    async def fetch_schema_slice_from_reasoning_engine(self, primary_classes: List[str]) -> Optional[Dict[str, Any]]:
        """Queries the Reasoning Engine REST API for dynamic schema axioms."""
        if not self.reasoning_engine_url:
            return None
        
        try:
            url = f"{self.reasoning_engine_url.rstrip('/')}/api/schema"
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json={"classes": primary_classes})
                if resp.status_code == 200:
                    logger.info(f"Retrieved dynamic schema slice from Reasoning Engine for classes: {primary_classes}")
                    return resp.json()
                else:
                    logger.warning(f"Reasoning Engine returned HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.warning(f"Failed to fetch schema slice from Reasoning Engine: {e}")
        return None

    def fetch_schema_slice_from_bigquery(self, primary_classes: List[str]) -> Optional[Dict[str, Any]]:
        """Queries BigQuery for class definitions and allowed properties."""
        client = self._get_bq_client()
        if not client or not primary_classes:
            return None
        
        query = f"""
            SELECT class_name, property_name, property_type, range_class
            FROM `{self.project_id}.{self.bq_dataset}.{self.bq_table}`
            WHERE class_name IN UNNEST(@classes)
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("classes", "STRING", primary_classes)
            ]
        )
        try:
            query_job = client.query(query, job_config=job_config)
            rows = list(query_job.result())
            if not rows:
                return None
            
            schema_slice: Dict[str, Any] = {}
            for row in rows:
                cname = row["class_name"]
                if cname not in schema_slice:
                    schema_slice[cname] = {
                        "class_name": cname,
                        "allowed_predicates": [],
                        "property_types": {}
                    }
                pname = row["property_name"]
                schema_slice[cname]["allowed_predicates"].append(pname)
                schema_slice[cname]["property_types"][pname] = row["range_class"] or row["property_type"]
            
            logger.info(f"Retrieved schema slice from BigQuery with {len(schema_slice)} classes")
            return schema_slice
        except Exception as e:
            logger.warning(f"BigQuery schema query failed: {e}")
            return None

    async def get_schema_slice(self, primary_classes: List[str]) -> Dict[str, Any]:
        """Dynamically retrieves schema slices across Reasoning Engine -> BigQuery -> domain registry."""
        # 1. Try Reasoning Engine API
        re_schema = await self.fetch_schema_slice_from_reasoning_engine(primary_classes)
        if re_schema:
            return re_schema
        
        # 2. Try BigQuery
        bq_schema = self.fetch_schema_slice_from_bigquery(primary_classes)
        if bq_schema:
            return bq_schema
        
        # 3. Dynamic slice from domain ontology registry based on detected primary classes
        schema_slice = {}
        for cls in primary_classes:
            if cls in DEFAULT_DOMAIN_SCHEMAS:
                schema_slice[cls] = DEFAULT_DOMAIN_SCHEMAS[cls]
            else:
                # Dynamically construct open-world schema representation for novel detected class
                schema_slice[cls] = {
                    "class_name": cls,
                    "description": f"Domain entity of type {cls}",
                    "allowed_predicates": ["hasProperty", "hasValue", "relatedTo", "partOf", "subClassOf"],
                    "property_types": {
                        "hasProperty": "xsd:string",
                        "hasValue": "QuantitativeValue",
                        "relatedTo": "owl:Thing",
                        "partOf": "owl:Thing"
                    }
                }
        
        if not schema_slice:
            schema_slice = DEFAULT_DOMAIN_SCHEMAS
        
        return schema_slice

    def format_schema_for_prompt(self, schema_slice: Dict[str, Any]) -> str:
        """Formats the retrieved schema slice into concise, strict prompt guidelines."""
        lines = ["=== DYNAMIC ONTOLOGY SCHEMA SLICE ==="]
        for cls_name, cls_data in schema_slice.items():
            lines.append(f"Class: {cls_name}")
            if "description" in cls_data:
                lines.append(f"  Description: {cls_data['description']}")
            preds = cls_data.get("allowed_predicates", [])
            types = cls_data.get("property_types", {})
            lines.append("  Allowed Predicates & Target Types:")
            for p in preds:
                t = types.get(p, "owl:Thing")
                lines.append(f"    - {p} -> {t}")
        lines.append("=====================================")
        return "\n".join(lines)
