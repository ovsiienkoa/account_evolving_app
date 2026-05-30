import json
from google.cloud import bigquery
from google import genai
from google.genai import types
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import get_analyze_history_prompt, get_process_custom_prompt

class ReplayAgent:
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

    def fetch_unprocessed_history(self) -> list:
        """
        Fetches history records where status is False (unprocessed).
        """
        if not self.bq_client:
            return []
            
        query = f"""
            SELECT entry_id, user_request, performed_actions 
            FROM `{self.project_id}.{self.bq_dataset}.history`
            WHERE status = False
            LIMIT 10
        """
        try:
            job = self.bq_client.query(query)
            return [dict(row) for row in job.result()]
        except Exception as e:
            print(f"Error fetching history: {e}")
            return []

    def analyze_history(self, history_rows: list) -> list:
        """
        Analyzes the history rows to determine what functionality is lacking, 
        and formulates a plan of action.
        """
        if not history_rows:
            return []
            
        schema_context = self._gather_context()
        
        prompt = get_analyze_history_prompt(
            schema_context=schema_context,
            project_id=self.project_id,
            bq_dataset=self.bq_dataset,
            history_rows_json=json.dumps(history_rows, indent=2)
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
            print(f"Error analyzing history: {e}")
            return []

    def process_custom_prompt(self, user_prompt: str) -> list:
        """
        Analyzes a custom user prompt to determine what functionality is lacking, 
        and formulates a plan of action.
        """
        if not user_prompt:
            return []
            
        schema_context = self._gather_context()
        
        prompt = get_process_custom_prompt(
            schema_context=schema_context,
            user_prompt=user_prompt,
            project_id=self.project_id,
            bq_dataset=self.bq_dataset
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
            print(f"Error processing custom prompt: {e}")
            return []

    def generate_and_execute_ddl(self, plan: list) -> bool:
        """
        Executes the DDL statements proposed by the LLM.
        """
        if not plan or not self.bq_client:
            return False
            
        success = True
        for item in plan:
            ddl = item.get("ddl_statement", "")
            if not ddl:
                continue
            
            print(f"Executing DDL:\n{ddl}\n")
            error_counter = 0
            success_flag = False
            current_ddl = ddl
            
            while error_counter < 3 and not success_flag:
                try:
                    job = self.bq_client.query(current_ddl)
                    job.result()  # Wait for query to complete
                    print("DDL executed successfully.")
                    success_flag = True
                except Exception as e:
                    print(f"Error executing DDL: {e}. Retrying... ({error_counter + 1}/3)")
                    error_counter += 1
                    if error_counter < 3:
                        fix_prompt = f"The following BigQuery DDL statement failed to execute.\n\nStatement:\n```sql\n{current_ddl}\n```\n\nError:\n{e}\n\nPlease fix the statement and return ONLY the corrected valid BigQuery DDL string. Do not use markdown blocks."
                        try:
                            response = self.client.models.generate_content(
                                model=self.model_name,
                                contents=fix_prompt
                            )
                            current_ddl = response.text.strip()
                            if current_ddl.startswith("```sql"):
                                current_ddl = current_ddl[6:]
                            if current_ddl.startswith("```"):
                                current_ddl = current_ddl[3:]
                            if current_ddl.endswith("```"):
                                current_ddl = current_ddl[:-3]
                            current_ddl = current_ddl.strip()
                        except Exception as inner_e:
                            print(f"Failed to generate fix: {inner_e}")
                            break
                    else:
                        success = False
                
        return success

    def mark_as_processed(self, entry_ids: list) -> bool:
        """
        Updates the history table to mark the specified entry_ids as processed (status = True).
        """
        if not entry_ids or not self.bq_client:
            return False
            
        formatted_ids = ", ".join([f"'{eid}'" for eid in entry_ids])
        query = f"""
            UPDATE `{self.project_id}.{self.bq_dataset}.history`
            SET status = True
            WHERE entry_id IN ({formatted_ids})
        """
        try:
            job = self.bq_client.query(query)
            job.result()
            print(f"Marked {len(entry_ids)} rows as processed.")
            return True
        except Exception as e:
            print(f"Error updating history status: {e}")
            return False
