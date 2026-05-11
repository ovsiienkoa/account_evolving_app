# Receipt Processing Agent

The `receipt-processing-agent` is responsible for extracting structured information from user-provided receipts.

## Workflow
1. Takes an image of a receipt from the user.
2. Calls Google Cloud Document AI generic OCR processor to extract a raw text representation of the image.
3. Uses a large language model (LLM) to parse this unstructured text into a structured JSON format conforming to the database schema.
4. Normalizes item names to ensure consistency in the database.

## Usage
The agent is designed to be instantiated as a class and passed necessary GCP configuration keys:
```python
from agent import ReceiptProcessingAgent

config = {
    "GCP_PROJECT_ID": "your-project-id",
    "GCP_LOCATION": "us",
    "GCP_PROCESSOR_ID": "your-processor-id"
}

agent = ReceiptProcessingAgent(config)
result = agent.process_receipt(image_bytes, mime_type="image/jpeg")
```
