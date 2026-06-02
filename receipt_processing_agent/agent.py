from google.cloud import documentai
from google.api_core.client_options import ClientOptions
import json
import hashlib
import copy
from google import genai
from google.genai import types
from google.cloud import bigquery
from pydantic import BaseModel, Field
from typing import List, Optional
import sys
import os
import io
from pypdf import PdfReader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import get_extract_receipt_data_prompt, get_normalize_names_prompt, get_summarize_receipt_prompt, get_create_brief_description_prompt, get_fix_receipt_data_prompt


class ReceiptItem(BaseModel):
    #id: str = Field(description="The unique identifier of the item.")
    full_name: str = Field(description="The exact name of the product from the receipt.")
    price: float = Field(description="The total price for this item.")
    currency: str = Field(description="The 3-letter currency code.")
    uom: Optional[str] = Field(default="None", description="The unit of measurement (e.g., kg, l, pcs, packs) if available.")
    measurement: Optional[float] = Field(default=1, description="The numeric measurement value (e.g., 1.5, 500, 1) corresponding to the unit of measurement.")

class ReceiptExtraction(BaseModel):
    transaction_date: str = Field(description="The date of the transaction in YYYY-MM-DD format.")
    merchant: Optional[str] = Field(description="The name of the store or merchant.")
    total_amount: float = Field(description="The total amount paid.")
    currency: str = Field(description="The 3-letter currency code for the total amount.")
    items: List[ReceiptItem] = Field(description="List of items purchased.")

class NormalizedNameMapping(BaseModel):
    original_name: str
    normalized_name: str = Field(description="Translated to English, stripped of brands, weights, and specifications.")

class NormalizationResult(BaseModel):
    mappings: List[NormalizedNameMapping]

DEBUG = False

class ReceiptProcessingAgent:
    def __init__(self, config: dict):
        self.project_id = config.get("GCP_PROJECT_ID", "")
        self.location = config.get("GCP_LOCATION", "")
        self.vertex_location = config.get("DUMB_MODEL_LOCATION", "")
        self.processor_id = config.get("GCP_PROCESSOR_ID", "")
        self.bq_dataset = config.get("BQ_DATASET", "")
        assert all([self.project_id, self.location, self.processor_id, self.vertex_location]), "No GCP credentials provided"

        self.model_name = "gemini-2.5-flash"
        self.embeding_model_name = 'gemini-embedding-001'
        self.embeding_config = types.EmbedContentConfig(
            output_dimensionality=768,
            task_type="CLUSTERING"
        )
        
        self.client = genai.Client(vertexai=True, project=self.project_id, location=self.vertex_location)
        self.bq_client = bigquery.Client(project=self.project_id) if self.project_id else None
        opts = ClientOptions(api_endpoint=f"{self.location}-documentai.googleapis.com")
        self.document_client = documentai.DocumentProcessorServiceClient(client_options=opts)

    def extract_text_from_image(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
        """
        Extracts raw text from an image using Google Cloud Document AI's generic OCR processor.
        """
        if DEBUG:
            return """
            debug_string_from_ocr"""
        else:
            if not all([self.project_id, self.location, self.processor_id]):
                # Return dummy text if credentials are not fully provided
                return "Dry run: Extracted text from image (No GCP credentials provided)"
            
            #opts = ClientOptions(api_endpoint=f"{self.location}-documentai.googleapis.com")
            #client = documentai.DocumentProcessorServiceClient(client_options=opts)
            
            name = self.document_client.processor_path(self.project_id, self.location, self.processor_id)
            
            raw_document = documentai.RawDocument(content=image_bytes, mime_type=mime_type)
            request = documentai.ProcessRequest(name=name, raw_document=raw_document)
            result = self.document_client.process_document(request=request)

            return result.document.text

    def _extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """
        Extracts raw text from a PDF file using pypdf.
        """

        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            text_list = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_list.append(page_text)
            return "\n".join(text_list)
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    def _extract_receipt_data(self, text: str) -> dict:
        schema = ReceiptExtraction.model_json_schema()
        prompt = get_extract_receipt_data_prompt(
            schema_json=json.dumps(schema),
            text=text
        )
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            return json.loads(response.text)
        except Exception as e:
            raise RuntimeError(f"Gemini API error during receipt data extraction: {e}")

    def _normalize_names(self, items: list) -> dict:
        names = [item["full_name"] for item in items]
        schema = NormalizationResult.model_json_schema()
        prompt = get_normalize_names_prompt(
            schema_json=json.dumps(schema),
            names_json=json.dumps(names, ensure_ascii=False)
        )
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            return json.loads(response.text)
        except Exception as e:
            raise RuntimeError(f"Gemini API error during name normalization: {e}")
        
    def _summarize_receipt(self, receipt_data: dict) -> str:
        prompt = get_summarize_receipt_prompt(
            receipt_data_json=json.dumps(receipt_data, ensure_ascii=False)
        )
        try:
            response = self.client.models.generate_content(model=self.model_name, contents=prompt)
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini API error during receipt summarization: {e}")

    def _create_brief_description(self, final_data: dict) -> str:
        prompt = get_create_brief_description_prompt(
            final_data_json=json.dumps(final_data, ensure_ascii=False)
        )
        try:
            response = self.client.models.generate_content(model=self.model_name, contents=prompt)
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini API error during description creation: {e}")


    def process_receipt(self, image_bytes: bytes = None, mime_type: str = None, text_input: str = None, user_id: str = None, files: List[dict] = None) -> dict:
        """
        Complete flow: Extract text -> Parse with LLM -> Normalize Names -> Summarize
        """
        if not DEBUG:
            text_repr = text_input + "\n" if text_input else ""
            
            all_files = []
            if files:
                all_files.extend(files)
            if image_bytes:
                all_files=[{"content": image_bytes, "mime_type": mime_type}]
                
            for file_info in all_files:
                f_bytes = file_info.get("content")
                f_mime = file_info.get("mime_type") or "image/jpeg"
                f_name = file_info.get("name", "Document")
                
                if f_mime == "application/pdf":
                    pdf_text = self._extract_text_from_pdf(f_bytes)
                    text_repr += f"\nExtracted data from PDF ({f_name}):\n{pdf_text}\n"
                elif f_mime.startswith("image/"):
                    image_text = self.extract_text_from_image(f_bytes, f_mime)
                    text_repr += f"\nExtracted data from image ({f_name}):\n{image_text}\n"
        else:
            text_repr = "" # Not used in debug

        if not DEBUG:
            if not getattr(self, 'client', None):
                return {"text": "Google GenAI is not initialized. Please provide GCP credentials in .env.", "data": {}}
            
            try:
                extraction = self._extract_receipt_data(text_repr)
                normalization = self._normalize_names(extraction.get("items", []))
                
                # Map normalized names back to items and add MD5 hashes
                mappings = {m["original_name"]: m["normalized_name"] for m in normalization.get("mappings", [])}
                
                final_items = []
                for item in extraction.get("items", []):
                    orig_name = item["full_name"]
                    norm_name = mappings.get(orig_name, orig_name) # fallback to original if missing
                    
                    # Create MD5 hash of normalized name
                    md5_hash = hashlib.md5(norm_name.encode('utf-8')).hexdigest()
                    
                    final_items.append({
                        "id": md5_hash,
                        "full_name": orig_name,
                        "normalized_name": norm_name,
                        "price": item["price"],
                        "currency": item["currency"],
                        "uom": item.get("uom"),
                        "measurement": item.get("measurement"),
                    })
                    
                final_data = {
                    "transaction_date": extraction.get("transaction_date"),
                    "user_id": user_id,
                    "merchant": extraction.get("merchant"),
                    "total_amount": extraction.get("total_amount"),
                    "currency": extraction.get("currency"),
                    "items": final_items
                }
            except Exception as e:
                return {
                    "text": f"Error processing receipt: {e}",
                    "data": {}
                }
        else:
            final_data = {
                'transaction_date': '2025-08-11',
                'user_id': user_id or '0',
                'merchant': 'ФОП Семененко С.А. ()',
                'total_amount': 646.8,
                'currency': 'UAH',
                'items': [
                    {
                        'id': '9c5192b6cb3a10cd92e7f421162891df',
                        'full_name': 'Трайфл Маракуя-Ананас 160г, шт Кондитер Дніпро',
                        'normalized_name': 'Trifle',
                        'price': 215.6,
                        'currency': 'UAH',
                        'uom': 'g',
                        'measurement': 160.0
                    },
                    {
                        'id': '9c5192b6cb3a10cd92e7f421162891df',
                        'full_name': 'Трайфл Пінчер з Вишнею 160г, шт Кондитер Дніпро',
                        'normalized_name': 'Trifle',
                        'price': 215.6,
                        'currency': 'UAH',
                        'uom': 'g',
                        'measurement': 160.0
                    },
                    {
                        'id': '9c5192b6cb3a10cd92e7f421162891df',
                        'full_name': 'Трайфл Солона карамель 160г, шт Кондитер Дніпро',
                        'normalized_name': 'Trifle',
                        'price': 215.6,
                        'currency': 'UAH',
                        'uom': 'g',
                        'measurement': 160.0
                    }
                ]
            }
        
        try:
            readable_final_data = copy.deepcopy(final_data)
            if 'user_id' in readable_final_data:
                readable_final_data.pop('user_id')
                
            readable_final_data['items'] = [
                {
                    "full_name": item["full_name"],
                    "normalized_name": item.get("normalized_name"),
                    "price": item["price"],
                    "currency": item["currency"],
                    "uom": item.get("uom"),
                    "measurement": item.get("measurement"),
                }
                for item in readable_final_data['items']
            ]
            
            if text_input or not DEBUG:
                summary = self._summarize_receipt(readable_final_data)
            else:
                summary = """
                Here's a summary of your receipt:

                **Receipt Summary**

                *   **Merchant:** ФОП Семененко С.А. ()
                *   **Date:** 2025-08-11
                *   **Total Amount:** 646.80 UAH

                **Items Purchased:**

                | Item Name (or Normalized Name) | Unit of Measurement (uom) | Measurement Value | Count / Quantity | Unit Price (UAH) | Total Price (UAH) |
                | :----------------------------- | :------------------------ | :---------------- | :--------------- | :--------------- | :---------------- |
                | Trifle                         | g                         | 160.0             | 3                | 215.60           | 646.80            |
                """

            return {
                "text": summary,
                "data": final_data
            }
        except Exception as e:
            return {
                "text": f"Error during receipt summarization: {e}",
                "data": final_data
            }

    def fix_receipt_data(self, final_data: dict, feedback: str) -> dict:
        """
        Uses LLM to amend the parsed receipt data based on user feedback without reprocessing OCR.
        """
        try:
            schema = ReceiptExtraction.model_json_schema()
            readable_final_data = copy.deepcopy(final_data)
            if 'user_id' in readable_final_data:
                readable_final_data.pop('user_id')
            prompt = get_fix_receipt_data_prompt(
                readable_final_data_json=json.dumps(readable_final_data, ensure_ascii=False),
                feedback=feedback,
                schema_json=json.dumps(schema)
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            corrected_data = json.loads(response.text)
            
            # We need to re-run normalization for the corrected items
            normalization = self._normalize_names(corrected_data.get("items", []))
            mappings = {m["original_name"]: m["normalized_name"] for m in normalization.get("mappings", [])}
            
            final_items = []
            for item in corrected_data.get("items", []):
                orig_name = item["full_name"]
                norm_name = mappings.get(orig_name, orig_name)
                md5_hash = hashlib.md5(norm_name.encode('utf-8')).hexdigest()
                final_items.append({
                    "id": md5_hash,
                    "full_name": orig_name,
                    "normalized_name": norm_name,
                    "price": item["price"],
                    "currency": item["currency"],
                    "uom": item.get("uom"),
                    "measurement": item.get("measurement"),
                })
                
            final_data_corrected = {
                "transaction_date": corrected_data.get("transaction_date"),
                "user_id": final_data.get("user_id"),
                "merchant": corrected_data.get("merchant"),
                "total_amount": corrected_data.get("total_amount"),
                "currency": corrected_data.get("currency"),
                "items": final_items
            }
            
            readable_final_data = copy.deepcopy(final_data_corrected)
            if 'user_id' in readable_final_data:
                readable_final_data.pop('user_id')
                
            readable_final_data['items'] = [
                {
                    "full_name": item["full_name"],
                    "normalized_name": item.get("normalized_name"),
                    "price": item["price"],
                    "currency": item["currency"],
                    "uom": item.get("uom"),
                    "measurement": item.get("measurement"),
                }
                for item in readable_final_data['items']
            ]
            summary = self._summarize_receipt(readable_final_data)
            
            return {
                "text": summary,
                "data": final_data_corrected
            }
        except Exception as e:
            return {
                "text": f"Error fixing receipt data: {e}",
                "data": final_data
            }

    def commit_receipt(self, final_data: dict) -> bool:
        """
        Generates embeddings for new unique items and the full receipt,
        then inserts them into the BigQuery tables.
        """
        if not self.bq_client or not self.bq_dataset:
            print("BigQuery not initialized. Skipping commit.")
            return False
            
        unique_items_table = f"{self.project_id}.{self.bq_dataset}.unique_items"
        
        item_ids = [item["id"] for item in final_data.get("items", [])]
        existing_ids = set()
        
        if item_ids:
            query = f"SELECT id FROM `{unique_items_table}` WHERE id IN UNNEST(@item_ids)"
            job_config = bigquery.QueryJobConfig(
                query_parameters=[bigquery.ArrayQueryParameter("item_ids", "STRING", item_ids)]
            )
            try:
                results = self.bq_client.query(query, job_config=job_config)
                for row in results:
                    existing_ids.add(row.id)
            except Exception as e:
                print(f"BigQuery Error checking unique_items: {e}")
                
        for item in final_data.get("items", []):
            item_id = item["id"]
            norm_name = item.get("normalized_name", item["full_name"])
            
            if item_id not in existing_ids:
                try:
                    embed_res = self.client.models.embed_content(
                        model=self.embeding_model_name, 
                        contents=norm_name,
                        config=self.embeding_config,
                    )
                    embedding = embed_res.embeddings[0].values # [0] because API returns list of embeddings for batch requests
                    
                    rows_to_insert = [{"id": item_id, "name": norm_name, "embedding": embedding}]
                    errors = self.bq_client.insert_rows_json(unique_items_table, rows_to_insert)
                    if errors:
                        print(f"Error inserting unique item: {errors}")
                except Exception as e:
                    print(f"BigQuery Error on unique_items insert: {e}")
                
        # Handle main table
        main_table = f"{self.project_id}.{self.bq_dataset}.main"
        receipt_str = self._create_brief_description(final_data)
        try:
            embed_res = self.client.models.embed_content(
                model=self.embeding_model_name, 
                contents=receipt_str,
                config=self.embeding_config,
            )
            receipt_embedding = embed_res.embeddings[0].values # [0] because API returns list of embeddings for batch requests
            
            row = {
                "transaction_date": final_data.get("transaction_date"),
                "user_id": final_data.get("user_id"),
                "merchant": final_data.get("merchant"),
                "total_amount": float(final_data.get("total_amount")),
                "currency": final_data.get("currency"),
                "items": [
                    {
                        "id": i["id"],
                        "full_name": i["full_name"],
                        "price": float(i["price"]),
                        "currency": i["currency"],
                        "uom": i.get("uom"),
                        "measurement": float(i["measurement"]) if i.get("measurement") is not None else None
                    } for i in final_data.get("items", [])
                ],
                "receipt_embedding": receipt_embedding
            }
            
            errors = self.bq_client.insert_rows_json(main_table, [row])
            if errors:
                print(f"Error inserting main receipt: {errors}")
                return False
        except Exception as e:
            print(f"BigQuery Error on main insert: {e}")
            return False
            
        return True
