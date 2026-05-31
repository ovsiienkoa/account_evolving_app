import os
import argparse
from dotenv import load_dotenv
from replay_agent.agent import ReplayAgent

def main():
    parser = argparse.ArgumentParser(description="Run the Replay Agent.")
    parser.add_argument("-m", "--message", type=str, help="Custom prompt message for the agent.")
    parser.add_argument("-p", "--prompt_file", type=str, help="Path to a text file containing the custom prompt.")
    args = parser.parse_args()

    print("Starting Replay Agent...")
    
    # Load environment variables
    load_dotenv()
    
    config = {
        "GCP_PROJECT_ID": os.getenv("GCP_PROJECT_ID"),
        "SMART_MODEL_LOCATION": os.getenv("SMART_MODEL_LOCATION"),
        "BQ_DATASET": os.getenv("BQ_DATASET"),
    }
    
    try:
        agent = ReplayAgent(config)
    except AssertionError as e:
        print(f"Configuration Error: {e}")
        return
        
    custom_prompt = None
    if args.message:
        custom_prompt = args.message
    elif args.prompt_file:
        try:
            with open(args.prompt_file, 'r') as f:
                custom_prompt = f.read()
        except Exception as e:
            print(f"Error reading prompt file: {e}")
            return

    if custom_prompt:
        print("Processing custom prompt...")
        plan = agent.process_custom_prompt(custom_prompt)
        
        if plan:
            print(f"Found {len(plan)} actions to perform. Generating and executing DDL...")
            agent.generate_and_execute_ddl(plan)
        else:
            print("No actions to perform based on the prompt.")
            
        print("Replay Agent finished execution.")
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
