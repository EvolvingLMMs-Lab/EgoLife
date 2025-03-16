import argparse
import os

from egorag.agents.RagAgent import RagAgent
from egorag.database.Chroma import Chroma
from egorag.utils.gen_event import gen_event


# Set Args
def parse_args():
    parser = argparse.ArgumentParser(
        description="Initialize RagAgent with database and JSON path."
    )
    parser.add_argument("--db_name", required=True, help="Name of the Chroma database")
    parser.add_argument(
        "--json_path", required=True, help="Path to the JSON file for database creation"
    )
    parser.add_argument("--diary_dir", required=False,default="events_diary", help="Path to the diary directory")
    return parser.parse_args()


# Main function
def main():
    # Parse command line arguments
    args = parse_args()

    # Initialize the Chroma database with the provided name
    db_t = Chroma(name=args.db_name)

    # Initialize the RagAgent
    agent = RagAgent(database_t=db_t, name="A1_JAKE", video_base_dir="Egolife/train")

    # Create the database from the provided JSON file
    agent.create_database_from_json(args.json_path)

    gen_event(args.db_name, args.diary_dir)
    # Optional: View the content of the database (first 10 entries)
    agent.database_t.view_database(n=10)  # View the first 10 entries


# Entry point for the script
if __name__ == "__main__":
    main()
