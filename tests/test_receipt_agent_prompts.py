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

from prompts import get_summarize_receipt_prompt
from receipt_processing_agent.agent import ReceiptProcessingAgent

class TestReceiptAgentPrompts(unittest.TestCase):
    def test_get_summarize_receipt_prompt_contains_instructions(self):
        prompt = get_summarize_receipt_prompt('{"items": []}')
        self.assertIn("counts (quantity)", prompt)
        self.assertIn("unit of measurement", prompt)
        self.assertIn("measurement value", prompt)
        self.assertIn("grouping items by their name", prompt)

    @patch('receipt_processing_agent.agent.ReceiptProcessingAgent._extract_receipt_data')
    @patch('receipt_processing_agent.agent.ReceiptProcessingAgent._normalize_names')
    @patch('receipt_processing_agent.agent.ReceiptProcessingAgent._summarize_receipt')
    def test_process_receipt_includes_normalized_name_in_readable_data(
        self, mock_summarize, mock_normalize, mock_extract
    ):
        config = {
            "GCP_PROJECT_ID": "test-project",
            "GCP_LOCATION": "us",
            "VERTEX_LOCATION": "us-central1",
            "DUMB_MODEL_LOCATION": "us-central1",
            "GCP_PROCESSOR_ID": "test-processor",
            "BQ_DATASET": "test-dataset"
        }
        agent = ReceiptProcessingAgent(config)
        agent.client = MagicMock()

        # Mock extracting and normalizing data
        mock_extract.return_value = {
            "transaction_date": "2026-06-02",
            "merchant": "Test Shop",
            "total_amount": 100.0,
            "currency": "USD",
            "items": [
                {"full_name": "Test Product 100g", "price": 50.0, "currency": "USD", "uom": "g", "measurement": 100.0},
                {"full_name": "Test Product 100g", "price": 50.0, "currency": "USD", "uom": "g", "measurement": 100.0}
            ]
        }
        mock_normalize.return_value = {
            "mappings": [
                {"original_name": "Test Product 100g", "normalized_name": "Test Product"}
            ]
        }
        mock_summarize.return_value = "Mocked Summary"

        # Under the hood, process_receipt will prepare readable_final_data and call _summarize_receipt
        with patch('receipt_processing_agent.agent.DEBUG', False):
            res = agent.process_receipt(text_input="some text")

        # Verify that _summarize_receipt was called with readable_final_data containing normalized_name
        mock_summarize.assert_called_once()
        called_arg = mock_summarize.call_args[0][0]
        self.assertIn("items", called_arg)
        for item in called_arg["items"]:
            self.assertEqual(item["normalized_name"], "Test Product")
            self.assertEqual(item["uom"], "g")
            self.assertEqual(item["measurement"], 100.0)

if __name__ == "__main__":
    unittest.main()
