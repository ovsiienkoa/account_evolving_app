import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock GCP, GenAI, and documentai client options
sys.modules['google.cloud'] = MagicMock()
sys.modules['google.cloud.documentai'] = MagicMock()
sys.modules['google'] = MagicMock()
sys.modules['google.genai'] = MagicMock()
sys.modules['google.api_core.client_options'] = MagicMock()

# Append project root to path
sys.path.append(".")

from receipt_processing_agent.agent import ReceiptProcessingAgent

class TestReceiptProcessingAgent(unittest.TestCase):
    def setUp(self):
        self.config = {
            "GCP_PROJECT_ID": "test-project",
            "GCP_LOCATION": "us",
            "VERTEX_LOCATION": "us-central1",
            "GCP_PROCESSOR_ID": "test-processor",
            "BQ_DATASET": "test-dataset"
        }
        # Instantiate agent
        self.agent = ReceiptProcessingAgent(self.config)
        
        # Mock the LLM and DocumentAI methods
        self.agent.client = MagicMock()
        self.agent.document_client = MagicMock()
        
        # Mock the helper methods called by process_receipt
        self.agent._extract_receipt_data = MagicMock(return_value={
            "transaction_date": "2026-05-30",
            "merchant": "Test Merchant",
            "total_amount": 100.0,
            "currency": "USD",
            "items": [
                {"full_name": "Test Item", "price": 100.0, "currency": "USD"}
            ]
        })
        self.agent._normalize_names = MagicMock(return_value={
            "mappings": [
                {"original_name": "Test Item", "normalized_name": "test item"}
            ]
        })
        self.agent._summarize_receipt = MagicMock(return_value="Summary content")
        self.agent._create_brief_description = MagicMock(return_value="Brief desc")

    @patch('receipt_processing_agent.agent.documentai.RawDocument')
    @patch('receipt_processing_agent.agent.documentai.ProcessRequest')
    def test_single_image_backwards_compatibility(self, mock_request, mock_raw_doc):
        # Mock documentai return text
        mock_result = MagicMock()
        mock_result.document.text = "OCR text from single image"
        self.agent.document_client.process_document.return_value = mock_result
        
        res = self.agent.process_receipt(
            image_bytes=b"fakeimagebytes",
            mime_type="image/png",
            user_id="user123"
        )
        
        # Verify OCR was called once
        self.agent.document_client.process_document.assert_called_once()
        # Verify LLM extraction was called with OCR text
        self.agent._extract_receipt_data.assert_called_once_with("\nExtracted data from image (Document):\nOCR text from single image\n")
        self.assertEqual(res["text"], "Summary content")
        self.assertEqual(res["data"]["total_amount"], 100.0)

    @patch('receipt_processing_agent.agent.documentai.RawDocument')
    @patch('receipt_processing_agent.agent.documentai.ProcessRequest')
    def test_multiple_images(self, mock_request, mock_raw_doc):
        mock_result1 = MagicMock()
        mock_result1.document.text = "OCR text 1"
        mock_result2 = MagicMock()
        mock_result2.document.text = "OCR text 2"
        self.agent.document_client.process_document.side_effect = [mock_result1, mock_result2]
        
        files = [
            {"content": b"bytes1", "mime_type": "image/png", "name": "img1.png"},
            {"content": b"bytes2", "mime_type": "image/jpeg", "name": "img2.jpg"}
        ]
        
        res = self.agent.process_receipt(
            files=files,
            user_id="user123"
        )
        
        # Verify OCR was called twice
        self.assertEqual(self.agent.document_client.process_document.call_count, 2)
        # Verify text representation contains text from both images
        extracted_text = self.agent._extract_receipt_data.call_args[0][0]
        self.assertIn("OCR text 1", extracted_text)
        self.assertIn("OCR text 2", extracted_text)
        self.assertIn("img1.png", extracted_text)
        self.assertIn("img2.jpg", extracted_text)

    @patch('pypdf.PdfReader')
    def test_pdf_processing(self, mock_pdf_reader):
        # Mock PdfReader behavior
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Text from PDF page"
        
        mock_reader_instance = MagicMock()
        mock_reader_instance.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader_instance
        
        files = [
            {"content": b"pdfbytes", "mime_type": "application/pdf", "name": "receipt.pdf"}
        ]
        
        res = self.agent.process_receipt(
            files=files,
            user_id="user123"
        )
        
        # Verify LLM extraction was called with PDF text
        extracted_text = self.agent._extract_receipt_data.call_args[0][0]
        self.assertIn("Text from PDF page", extracted_text)
        self.assertIn("receipt.pdf", extracted_text)

if __name__ == "__main__":
    unittest.main()
