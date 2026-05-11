import os
from dotenv import load_dotenv
from replay_agent.agent import ReplayAgent

def main():
    print("Starting Replay Agent...")
    
    # Load environment variables
    load_dotenv()
    
    config = {
        "GCP_PROJECT_ID": os.getenv("GCP_PROJECT_ID"),
        "VERTEX_LOCATION": os.getenv("VERTEX_LOCATION"),
        "BQ_DATASET": os.getenv("BQ_DATASET"),
    }
    
    try:
        agent = ReplayAgent(config)
    except AssertionError as e:
        print(f"Configuration Error: {e}")
        return
        
    # Step 1: Fetch unprocessed history
    print("Fetching unprocessed history...")
    history_rows = agent.fetch_unprocessed_history()
    
    if not history_rows:
        print("No unprocessed history found. Exiting.")
        return
        
    print(f"Found {len(history_rows)} unprocessed records.")
    
    # Step 2: Analyze history
    print("Analyzing history for missing functionality...")
    plan = agent.analyze_history(history_rows)
    
    # Step 3: Execute DDL if any improvements were found
    if plan:
        print(f"Found {len(plan)} potential improvements. Generating and executing DDL...")
        agent.generate_and_execute_ddl(plan)
    else:
        print("No schema improvements needed based on recent history.")
        
    # Step 4: Mark rows as processed
    print("Marking rows as processed...")
    entry_ids = [row["entry_id"] for row in history_rows if "entry_id" in row]
    if entry_ids:
        agent.mark_as_processed(entry_ids)
        
    print("Replay Agent finished execution.")

if __name__ == "__main__":
    main()
