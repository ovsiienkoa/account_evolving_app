# Data Analytics Agent

The `data_analytics_agent` translates natural language user queries into executable BigQuery SQL requests by dynamically reading the database schema.

## Workflow (Current Basic Phase)
1. Receives natural language questions from the user (e.g., "What was my top spending category last month?").
2. Connects to BigQuery to dynamically fetch the latest descriptions and schemas of all tables and User Defined Functions (UDFs) within the project's dataset.
3. Constructs an LLM prompt containing this live context alongside the user's question.
4. Generates a valid BigQuery SQL string capable of retrieving the requested data.

*(Note: Plot generation, automated query execution, and history tracking are planned for future phases.)*

## Usage
The agent requires standard GCP credentials defined in the environment config:
```python
from data_analytics_agent.agent import DataAnalyticsAgent

config = {
    "GCP_PROJECT_ID": "...",
    "VERTEX_LOCATION": "...",
    "BQ_DATASET": "..."
}

agent = DataAnalyticsAgent(config)
sql_query_result = agent.generate_sql("How much did I spend in August 2025?")
```
