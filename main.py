import sys
import argparse
import json
from orchestration.flow import RaggedyOrchestrator

def main():
    parser = argparse.ArgumentParser(description="RAGGEDY TOOL - Offline Knowledge Intelligence System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents from data/raw")
    
    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("query", type=str, help="The question to ask")

    # UI command
    ui_parser = subparsers.add_parser("ui", help="Launch the Streamlit UI")

    args = parser.parse_args()
    
    orchestrator = RaggedyOrchestrator()

    if args.command == "ingest":
        print("Ingesting new documents...")
        orchestrator.process_new_documents()
        print("Ingestion complete.")
    elif args.command == "ask":
        print(f"Querying: {args.query}")
        result = orchestrator.ask(args.query)
        print("\n--- ANSWER ---")
        print(result["answer"])
        print("\n--- SOURCES ---")
        for i, src in enumerate(result["sources"]):
            print(f"[{i+1}] {src['filename']} (Method: {src['method']})")
            # print(f"    {src['text'][:100]}...")
    elif args.command == "ui":
        import subprocess
        from utils.logger import raggedy_logger
        raggedy_logger.info("Launching UI...")
        subprocess.run(["streamlit", "run", "ui.py"])
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
