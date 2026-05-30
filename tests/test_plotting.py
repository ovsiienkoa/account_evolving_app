import sys
import unittest
from unittest.mock import MagicMock

# Mock out GCP and GenAI modules before importing DataAnalyticsAgent
sys.modules['google.cloud'] = MagicMock()
sys.modules['google'] = MagicMock()
sys.modules['google.genai'] = MagicMock()

# Append project root to path
sys.path.append(".")

from data_analytics_agent.agent import DataAnalyticsAgent
import plotly.graph_objects as go

class TestDataAnalyticsAgentPlotting(unittest.TestCase):
    def setUp(self):
        config = {
            "GCP_PROJECT_ID": "mock-project",
            "VERTEX_LOCATION": "us-central1",
            "SMART_MODEL_LOCATION": "us-central1",
            "BQ_DATASET": "mock-dataset"
        }
        self.agent = DataAnalyticsAgent(config)
        self.dummy_data = [
            {"date": "2026-05-25", "weekday_group": "Weekday", "amount": 10.0, "quantity": 1},
            {"date": "2026-05-26", "weekday_group": "Weekday", "amount": 15.5, "quantity": 2},
            {"date": "2026-05-27", "weekday_group": "Weekday", "amount": 8.0, "quantity": 1},
            {"date": "2026-05-30", "weekday_group": "Weekend", "amount": 30.0, "quantity": 3},
            {"date": "2026-05-31", "weekday_group": "Weekend", "amount": 45.0, "quantity": 5},
        ]

    def test_make_plot_empty_or_none(self):
        # Empty config
        self.assertIsNone(self.agent.make_plot({}, self.dummy_data))
        # Empty data
        self.assertIsNone(self.agent.make_plot({"type": "line", "x_col": "date", "y_col": "amount"}, []))
        # None data
        self.assertIsNone(self.agent.make_plot({"type": "line", "x_col": "date", "y_col": "amount"}, None))
        # None config
        self.assertIsNone(self.agent.make_plot(None, self.dummy_data))

    def test_make_plot_simple_line(self):
        config = {
            "type": "line",
            "x_col": "date",
            "y_col": "amount",
            "title": "Simple Line Chart",
            "x_label": "Date Axis",
            "y_label": "Amount Axis"
        }
        fig = self.agent.make_plot(config, self.dummy_data)
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, go.Figure)
        
        # Verify labels inside the figure data
        self.assertEqual(fig.layout.title.text, "Simple Line Chart")
        self.assertEqual(fig.layout.xaxis.title.text, "Date Axis")
        self.assertEqual(fig.layout.yaxis.title.text, "Amount Axis")

    def test_make_plot_colored_line(self):
        config = {
            "type": "line",
            "x_col": "date",
            "y_col": "amount",
            "color_col": "weekday_group",
            "title": "Colored Line Chart"
        }
        fig = self.agent.make_plot(config, self.dummy_data)
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(fig.layout.title.text, "Colored Line Chart")
        
        # In Plotly, specifying a color groups lines, creating multiple trace entries.
        # Let's verify we have traces for both 'Weekday' and 'Weekend'
        self.assertTrue(len(fig.data) >= 2)
        trace_names = {trace.name for trace in fig.data}
        self.assertIn("Weekday", trace_names)
        self.assertIn("Weekend", trace_names)

    def test_make_plot_stacked_bar(self):
        config = {
            "type": "bar",
            "x_col": "date",
            "y_col": "amount",
            "color_col": "weekday_group",
            "barmode": "stack",
            "title": "Stacked Bar Chart"
        }
        fig = self.agent.make_plot(config, self.dummy_data)
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(fig.layout.barmode, "stack")
        self.assertEqual(fig.layout.title.text, "Stacked Bar Chart")

    def test_make_plot_scatter(self):
        config = {
            "type": "scatter",
            "x_col": "amount",
            "y_col": "quantity",
            "color_col": "weekday_group",
            "size_col": "quantity",
            "hover_name": "date",
            "title": "Scatter Plot"
        }
        fig = self.agent.make_plot(config, self.dummy_data)
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(fig.layout.title.text, "Scatter Plot")

    def test_make_plot_pie(self):
        config = {
            "type": "pie",
            "x_col": "weekday_group",
            "y_col": "amount",
            "title": "Pie Chart"
        }
        fig = self.agent.make_plot(config, self.dummy_data)
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(fig.layout.title.text, "Pie Chart")
        
        # Verify slices data matches weekday groups
        labels = [slice_label for slice_label in fig.data[0].labels]
        self.assertIn("Weekday", labels)
        self.assertIn("Weekend", labels)

    def test_make_plot_histogram(self):
        config = {
            "type": "histogram",
            "x_col": "amount",
            "title": "Histogram of Amounts"
        }
        fig = self.agent.make_plot(config, self.dummy_data)
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, go.Figure)

    def test_make_plot_box(self):
        config = {
            "type": "box",
            "x_col": "weekday_group",
            "y_col": "amount",
            "title": "Box Plot"
        }
        fig = self.agent.make_plot(config, self.dummy_data)
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, go.Figure)

    def test_make_plot_area(self):
        config = {
            "type": "area",
            "x_col": "date",
            "y_col": "amount",
            "color_col": "weekday_group",
            "title": "Area Chart"
        }
        fig = self.agent.make_plot(config, self.dummy_data)
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, go.Figure)

    def test_make_plot_invalid_column_ignored(self):
        config = {
            "type": "line",
            "x_col": "date",
            "y_col": "amount",
            "color_col": "invalid_column_name_that_does_not_exist",
            "title": "Invalid Column Chart"
        }
        # Should not crash on invalid column but handle it gracefully
        fig = self.agent.make_plot(config, self.dummy_data)
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, go.Figure)
        # Verify color is not grouping since the column was ignored/filtered out
        # So we should only have one trace instead of multiple grouping traces
        self.assertEqual(len(fig.data), 1)

if __name__ == '__main__':
    unittest.main()
