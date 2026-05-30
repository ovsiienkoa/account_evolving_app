import json
import re
import plotly.graph_objects as go
import plotly.express as px
from google.cloud import bigquery
from google import genai
from google.genai import types
import hashlib
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import get_generate_sql_prompt, get_format_answer_prompt, get_commit_to_history_prompt

DEBUG = False

class DataAnalyticsAgent:
    def __init__(self, config: dict):
        self.project_id = config.get("GCP_PROJECT_ID", "")
        self.vertex_location = config.get("VERTEX_LOCATION", "")
        self.bq_dataset = config.get("BQ_DATASET", "")
        
        assert all([self.project_id, self.vertex_location, self.bq_dataset]), "Missing GCP credentials in config"
        
        self.client = genai.Client(vertexai=True, project=self.project_id, location=self.vertex_location)
        self.model_name = "gemini-2.5-flash"
        self.bq_client = bigquery.Client(project=self.project_id) if self.project_id else None

    def _gather_context(self) -> str:
        """
        Gathers schema information from BigQuery for all tables and routines in the dataset.
        """
        if not self.bq_client:
            return "BigQuery client is not initialized."
            
        context_parts = []
        context_parts.append(f"Database: BigQuery\nProject: {self.project_id}\nDataset: {self.bq_dataset}\n")
        
        dataset_ref = self.bq_client.dataset(self.bq_dataset)
        
        try:
            # Gather tables
            tables = list(self.bq_client.list_tables(dataset_ref))
            if tables:
                context_parts.append("### Tables ###\n")
                for table_item in tables:
                    table = self.bq_client.get_table(table_item.reference)
                    desc = table.description or "No description."
                    
                    def _parse_schema(fields, prefix=""):
                        s_str = ""
                        for field in fields:
                            full_name = f"{prefix}{field.name}"
                            s_str += f"  - {full_name}: {field.field_type} ({field.mode}) : {field.description or 'No description'}\n"
                            if field.field_type == "RECORD" and field.fields:
                                s_str += _parse_schema(field.fields, prefix=f"{full_name}.")
                        return s_str
                        
                    schema_str = _parse_schema(table.schema)
                        
                    context_parts.append(f"Table Name: {table.table_id}\nDescription: {desc}\nSchema:\n{schema_str}\n")
                    
            # Gather routines (UDFs)
            routines = list(self.bq_client.list_routines(dataset_ref))
            if routines:
                context_parts.append("### Functions (UDFs) ###\n")
                for routine_item in routines:
                    routine = self.bq_client.get_routine(routine_item.reference)
                    
                    args = []
                    for arg in (routine.arguments or []):
                        data_type = getattr(arg, 'data_type', '')
                        args.append(f"{arg.name} {data_type}")
                    args_str = ", ".join(args)
                    desc = routine.description or "No description."
                    
                    context_parts.append(f"Function Name: {routine.routine_id}\nArguments: ({args_str})\nDescription: {desc}\n\n")
                    
        except Exception as e:
            return f"Error gathering context from BigQuery: {e}"
            
        return "\n".join(context_parts)

    def generate_sql(self, user_query: str, user_id: str = None, error_feedback: str = None) -> dict:
        """
        Uses the gathered context and user query to generate a valid BigQuery SQL statement.
        """
        if not DEBUG:
            schema_context = self._gather_context()
            
            prompt = get_generate_sql_prompt(
                schema_context=schema_context,
                user_id=user_id,
                user_query=user_query,
                project_id=self.project_id,
                bq_dataset=self.bq_dataset
            )
            
            if error_feedback:
                prompt += error_feedback
                
            print("Generating SQL query...")

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            sql_query = response.text.strip()
        else:
            sql_query = """
            debug_sql_query
        """
        
        # Clean up common LLM artifacts if it still outputs markdown blocks
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.startswith("```"):
            sql_query = sql_query[3:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
            
        sql_query = sql_query.strip()
        
        return {
            "text": f"Generated SQL Query:\n```sql\n{sql_query}\n```",
            "data": {
                "generated_sql": sql_query
            }
        }

    def generate_and_execute_sql(self, user_query: str, user_id: str = None) -> tuple[dict, list]:
        """
        Runs generate_sql and execute_query in a retry loop.
        """
        error_counter = 0
        success_flag = False
        error_feedback = ""
        sql_output = []
        response_dict = {}
        
        while error_counter < 3 and not success_flag:
            print(f"Attempting SQL generation and execution (Attempt {error_counter + 1})")
            response_dict = self.generate_sql(user_query, user_id, error_feedback)
            sql_query = response_dict["data"]["generated_sql"]
            
            # Security review before execution
            self.sql_review(sql_query, user_id)
            
            try:
                sql_output = self.execute_query(sql_query)
                success_flag = True
            except Exception as e:
                print(f"Execution failed: {e}")
                error_feedback += f"\n\nERROR: The query you generated failed to execute. Error: {e}\nQuery:\n{sql_query}\nPlease correct it."
                error_counter += 1
                
        if not success_flag:
            print("Failed to generate executable SQL after 3 attempts.")
            
        return response_dict, sql_output

    def sql_review(self, sql_query: str, user_id: str = None) -> None:
        """
        Reviews a SQL query for security risks before execution.
        1. Checks for DROP statements (case-insensitive word check).
        2. If user_id is provided, checks that the query filters by user_id = <user_id>.
        """
        if not sql_query:
            return

        # 1. DROP check (case-insensitive word check)
        if re.search(r'\bdrop\b', sql_query, re.IGNORECASE):
            raise ValueError("We suspect you of being naughty.")

        # 2. user_id check
        if user_id:
            # Match user_id = 'val' or user_id="val" or user_id=val
            # Allows table prefixes/aliases (e.g. t.user_id)
            matches = re.findall(r'\b(?:\w+\.)?user_id\b\s*=\s*(?:[\'"]([^\'"]+)[\'"]|([^\s;\'"]+))', sql_query, re.IGNORECASE)
            
            if not matches:
                raise ValueError("We suspect you of being naughty.")
                
            for m in matches:
                val = m[0] or m[1]
                if val != user_id:
                    raise ValueError("We suspect you of being naughty.")

    def execute_query(self, sql_query: str) -> list:
        if DEBUG:
            return [
                {"category": "Groceries", "total_spent": 120.50},
                {"category": "Entertainment", "total_spent": 45.00},
                {"category": "Transport", "total_spent": 30.00}
            ]
            
        if not self.bq_client:
            raise Exception("BigQuery client not initialized.")
            
        try:
            query_job = self.bq_client.query(sql_query)
            results = query_job.result()
            return [dict(row) for row in results]
        except Exception as e:
            print(f"Error executing query: {e}")
            if DEBUG: # Fallback if debug wasn't set but we want to fail gracefully
                return [{"error": str(e)}]
            raise e

    def format_answer(self, user_query: str, sql_query: str, sql_output: list, feedback: str = None) -> dict:
        prompt = get_format_answer_prompt(
            user_query=user_query,
            sql_query=sql_query,
            sql_output=sql_output,
            feedback=feedback
        )
        if not DEBUG:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            response_text = response.text
        else:
            response_text =  """{
                "text_response": "Analysis: Based on the provided data, your total spending across all categories is $195.5. This includes $120.5 on Groceries, $45.0 on Entertainment, and $30.0 on Transport.",
                "plot_config": null
            }"""
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return {"text_response": "Error parsing LLM output.", "plot_config": None}

    def make_plot(self, plot_config: dict, sql_output: list):
        if not plot_config or not sql_output:
            return None
            
        plot_type = plot_config.get("type", "bar")
        title = plot_config.get("title", "")
        x_col = plot_config.get("x_col", "")
        y_col = plot_config.get("y_col", "")
        
        # Extract data from sql_output based on x_col and y_col
        x = [row.get(x_col) for row in sql_output] if x_col else []
        y = [row.get(y_col) for row in sql_output] if y_col else []
        try:
            if plot_type == "bar":
                fig = px.bar(x=x, y=y, title=title, labels={"x": plot_config.get("x_label", ""), "y": plot_config.get("y_label", "")})
            elif plot_type == "pie":
                fig = px.pie(names=x, values=y, title=title)
            elif plot_type == "line":
                fig = px.line(x=x, y=y, title=title, labels={"x": plot_config.get("x_label", ""), "y": plot_config.get("y_label", "")})
            else:
                fig = px.bar(x=x, y=y, title=title)
        except Exception as e:
            print(f"Error generating plot: {e}")
            fig = None
            
        return fig

    def commit_to_history(self, user_query: str, sql_query: str, analysis: dict):
        """
        Commits the interaction story to the BigQuery history table.
        """
        if not self.bq_client:
            print("BigQuery client not initialized. Cannot commit to history.")
            return False
            
        prompt = get_commit_to_history_prompt(
            user_query=user_query,
            sql_query=sql_query
        )
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            reasoning = response.text.strip()
        except Exception as e:
            reasoning = f"Generated SQL query based on user prompt. Error generating reasoning: {e}"
            
        entry_id = hashlib.md5(user_query.encode('utf-8')).hexdigest()
        
        history_table = f"{self.project_id}.{self.bq_dataset}.history"
        row = {
            "entry_id": entry_id,
            "user_request": user_query,
            "performed_actions": reasoning,
            "status": False
        }
        
        try:
            errors = self.bq_client.insert_rows_json(history_table, [row])
            if errors:
                print(f"Error inserting into history: {errors}")
                return False
            print("Successfully committed interaction to history.")
            return True
        except Exception as e:
            print(f"BigQuery Error on history commit: {e}")
            return False
