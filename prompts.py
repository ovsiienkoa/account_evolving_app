import json

# ==========================================
# Data Analytics Agent Prompts
# ==========================================

def get_generate_sql_prompt(schema_context: str, user_id: str, user_query: str, project_id: str, bq_dataset: str) -> str:
    return f"""
            You are an expert Google BigQuery data analyst.
            Your task is to convert the user's natural language question into a valid BigQuery SQL statement.
            
            Here is the current live state of the project's database (Tables, schemas, and custom Functions/UDFs) pulled directly from BigQuery:
            ---
            {schema_context}
            ---
            User's ID in main database, all user could only see thei own data, if the table contains user_id in any from you absolutely should filter data by user_id select the next id:
            {user_id}
            User's Question:
            {user_query}
            
            Keep in mind, that you can perform a vector search on unique_items table or main table for getting exact ids of semantically appropriate items. 
            Example of vector search using AI.GENERATE_EMBEDDING:
            SELECT base.id, base.name, distance
            FROM VECTOR_SEARCH(
            TABLE `{project_id}.{bq_dataset}.unique_items`,
            'embedding',
            (SELECT embedding as embedding FROM AI.GENERATE_EMBEDDING(MODEL `{project_id}.{bq_dataset}.embedding_model`, (SELECT ‘product of interest text representation’ AS content),   STRUCT('CLUSTERING' AS task_type, 768 AS output_dimensionality))),
            top_k => 50
            )
            WHERE distance < 0.275

            Example of more efficient error proof query with use of TVF:
            SELECT * FROM `{project_id}.{bq_dataset}.search_unique_items`("product of interest text representation", 0.275)
            The last parameter is distance parameter, you can change it if needed.
            
            You should tweak the distance value, if you get semantically irrelevant items, try to decrease it, if you get no items, try to increase it. 
            Tweak distance value, if you get not all expected items.

            Example of getting exact objects from 'items' field in 'main' table
            SELECT
                item
            FROM
                `{project_id}.{bq_dataset}.main`,
            UNNEST(items) AS item

            Please output ONLY a valid BigQuery SQL query. Do not wrap it in markdown code blocks (e.g. do not use ```sql). 
            Do not explain the query. Provide just the raw string of the query itself.
            """

def get_format_answer_prompt(user_query: str, sql_query: str, sql_output: list, feedback: str = None) -> str:
    prompt = f"""
        You are an expert data analyst. You need to answer the user's question based on the SQL query that was run and the resulting output data.
        
        User's Question: {user_query}
        SQL Query Executed: {sql_query}
        SQL Output Data (First 5 rows): {json.dumps(sql_output[:5], default=str)}
        """
    if feedback:
        prompt += f"\nUser Feedback on previous analysis: {feedback}\nPlease adjust your analysis according to this feedback."
        
    prompt += """
        
        Respond with a JSON object exactly matching this schema:
        {
            "text_response": "A friendly, clear summary answering the user's question based on the data.",
            "plot_config": null  // Set to null if a plot is not useful or there's not enough data.
        }
        
        If a plot would be helpful (e.g. for comparing categories, showing a trend, grouping data, or correlation), replace `null` with:
        {
            "type": "bar" | "pie" | "line" | "scatter" | "histogram" | "box" | "area",
            "title": "Title of the chart",
            "x_col": "column_name_for_x_axis",  // The column name from SQL Output to use for the X-axis (or pie slices, box categories, etc.)
            "y_col": "column_name_for_y_axis",  // The column name from SQL Output to use for numerical values (e.g. height of bars, line points, pie values)
            "x_label": "Label for X axis",
            "y_label": "Label for Y axis",
            "color_col": "column_name_for_color", // Optional: The column name from SQL Output to group/color-code data by (e.g. weekday group, category)
            "size_col": "column_name_for_size", // Optional: The column name for marker sizes (useful for scatter plots)
            "hover_name": "column_name_for_hover_labels", // Optional: The column name to use for hover tooltips/titles
            "barmode": "group" | "stack" | "relative", // Optional: How bars are positioned in "bar" chart (defaults to "group")
            "facet_row": "column_name_for_facet_row", // Optional: The column name to split plots vertically into subplots
            "facet_col": "column_name_for_facet_col" // Optional: The column name to split plots horizontally into subplots
        }
        """
    return prompt

def get_commit_to_history_prompt(user_query: str, sql_query: str) -> str:
    return f"""
        Analyze this interaction:
        User Question: {user_query}
        SQL Solution: {sql_query}
        
        Provide a concise 1-2 sentence explanation of WHY this SQL query is the correct solution and WHAT kind of solution it is (e.g., vector search, aggregation, grouping).
        """

# ==========================================
# Receipt Processing Agent Prompts
# ==========================================

def get_extract_receipt_data_prompt(schema_json: str, text: str) -> str:
    return f"""
        Extract the following receipt text into a structured JSON format matching this schema:
        {schema_json}
        
        Receipt Text:
        {text}
        """

def get_normalize_names_prompt(schema_json: str, names_json: str) -> str:
    return f"""
        You are a product name normalization expert.
        I will give you a list of product names from a receipt.
        For each product name, you must:
        1. Translate it to English.
        2. Strip all brands.
        3. Strip all weights and measurements (e.g. 1kg, 500ml, size L).
        4. Strip any irrelevant specifications.
        For example, "Monitor Samsung 24 inch" -> "Monitor", "Coffee on oat milk 200ml" -> "Coffee", "Трайфл Маракуя-Ананас 160г, шт Кондитер Дніпро" -> "Trifle".
        
        Output MUST be a JSON matching this schema:
        {schema_json}
        
        Product Names to normalize:
        {names_json}
        """

def get_summarize_receipt_prompt(receipt_data_json: str) -> str:
    return f"""
        You are an accounting assistant. Please create a friendly, well-formatted Markdown summary of the following receipt.
        Include the Merchant, Date, Total Amount, and a markdown table of the items (showing their name and price).
        
        Receipt Data:
        {receipt_data_json}
        """

def get_create_brief_description_prompt(final_data_json: str) -> str:
    return f"""
        Provide a concise, 1-2 sentence description of this receipt, mentioning the merchant, total amount, and general category of items. Maybe additional useful information

        Receipt Data:
        {final_data_json}
        """

def get_fix_receipt_data_prompt(readable_final_data_json: str, feedback: str, schema_json: str) -> str:
    return f"""
        You are an accounting assistant. The user rejected the parsed receipt data and provided feedback.
        Please apply their feedback to correct the structured JSON.
        
        Current Receipt Data:
        {readable_final_data_json}
        
        User Feedback:
        {feedback}
        
        Output MUST match this schema exactly:
        {schema_json}
        """

# ==========================================
# Replay Agent Prompts
# ==========================================

def get_analyze_history_prompt(schema_context: str, project_id: str, bq_dataset: str, history_rows_json: str) -> str:
    return f"""
        You are an expert Google BigQuery Database Administrator.
        Your task is to analyze recent user requests and the actions taken by our data analytics agent to fulfill them.
        
        Current Database Schema and Functions:
        ---
        {schema_context}
        ---
        
        Note regarding tables:
        - The `main` table contains a RECORD REPEATED field `items`. To access items, you must UNNEST(items).
        - To match products semantically, we use BigQuery VECTOR_SEARCH on `unique_items` with embeddings generated via AI.GENERATE_EMBEDDING.
        
        Recent Interactions (Unprocessed):
        {history_rows_json}
        
        Analyze these interactions. For each query, write an extremely short description of what functionality (e.g., a specific View, an aggregation Table, or a BigQuery UDF) is lacking that would make answering this query easier, faster, or less error-prone for the data analytics agent.
        
        If there is no lacking functionality and the query was easily answered with the current schema, ignore it.
        
        Respond with a JSON array where each object has:
        - "reasoning": "Extremely short description of the lacking functionality"
        - "ddl_statement": "The exact CREATE OR REPLACE TABLE/VIEW or CREATE OR REPLACE FUNCTION statement to implement this."
        - "description": "A description to add to the table/function options, so the data-analytics-agent is aware of its existence."
        
        CRITICAL RULES for ddl_statement:
        - The statement MUST include OPTIONS(description="YOUR_DESCRIPTION_HERE") so the Data Analytics Agent automatically picks it up during introspection. This statement right after CREATE FUNCTION statement before AS ...
        - Ensure any tables/functions are created inside `{project_id}.{bq_dataset}`.
        - If no improvements are needed for ANY row, return an empty array `[]`.
        """

def get_process_custom_prompt(schema_context: str, user_prompt: str, project_id: str, bq_dataset: str) -> str:
    return f"""
        You are an expert Google BigQuery Database Administrator.
        Your task is to analyze a user request and formulate an action plan.
        
        Current Database Schema and Functions:
        ---
        {schema_context}
        ---
        
        User Request:
        {user_prompt}
        
        Analyze this request. Create a plan of action consisting of BigQuery DDL statements (e.g., CREATE TABLE, CREATE VIEW, CREATE FUNCTION).
        
        Respond with a JSON array where each object has:
        - "reasoning": "Extremely short description of why this is being created"
        - "ddl_statement": "The exact CREATE OR REPLACE TABLE/VIEW or CREATE OR REPLACE FUNCTION statement to implement this."
        - "description": "A description to add to the table/function options."
        
        CRITICAL RULES for ddl_statement:
        - The statement MUST include OPTIONS(description="YOUR_DESCRIPTION_HERE") so the Data Analytics Agent automatically picks it up during introspection. This statement right after CREATE FUNCTION statement before AS ...
        - Ensure any tables/functions are created inside `{project_id}.{bq_dataset}`.
        - If no action is needed, return an empty array `[]`.
        """
