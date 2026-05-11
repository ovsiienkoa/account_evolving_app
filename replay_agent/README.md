# Replay Agent

The `replay_agent` acts as an autonomous database optimizer and maintainer. It runs asynchronously (typically via cron job or manual trigger) rather than interacting with the user directly.

## Workflow (Phase 4)
1. **Fetch History**: It queries the `history` table to find recent interactions between the user and the Data Analytics agent that have not yet been analyzed (`status = False`).
2. **Analyze Shortcomings**: It uses an LLM to review the SQL queries that were generated and the actions taken. It identifies repetitive, complex, or lacking functionality in the current schema (e.g. repeated need for a specific `GROUP BY` grouping).
3. **Generate DDL**: If the agent decides a new View, Table, or User Defined Function (UDF) would simplify future querying, it generates the appropriate `CREATE OR REPLACE` BigQuery DDL. It enforces adding `OPTIONS(description="...")` to the DDL.
4. **Execute**: It executes the DDL directly against BigQuery, instantiating the new logic. Because it includes descriptions, the Data Analytics Agent will automatically pick up this new functionality on its next run.
5. **Mark as Processed**: It updates the analyzed rows in the `history` table, setting `status = True`.

## Usage
The agent requires standard GCP credentials defined in the environment config:

To trigger the agent, run the entrypoint script located at the root of the project:
```bash
python run_replay_agent.py
```
