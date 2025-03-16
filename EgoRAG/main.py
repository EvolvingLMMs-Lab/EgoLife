import argparse
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path

import yaml
from egorag.agents.RagAgent import RagAgent, call_gpt4
from egorag.database.Chroma import Chroma
from egorag.models import import_model
from egorag.utils.util import *
from egorag.utils.gen_event import gen_event
from tqdm import tqdm


def main(args):
    # Check if required files and directories exist
    if not os.path.exists(args.video_dir):
        raise FileNotFoundError(f"Video directory not found: {args.video_dir}")
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
        
    if not os.path.exists(args.query_json):
        raise FileNotFoundError(f"Query JSON file not found: {args.query_json}")

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        raise ValueError(f"Error loading config file: {e}")

    NAME = args.name
    DATE = args.date
    DB_NAME = args.db_name
    RUN_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract query_json_name after verifying file exists
    query_json_name = os.path.splitext(os.path.basename(args.query_json))[0]
    
    # Load query data
    try:
        with open(args.query_json, "r", encoding="utf-8") as json_file:
            query_data = json.load(json_file)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in query file: {e}")
    except Exception as e:
        raise Exception(f"Error reading query file: {e}")
        
    print(f"Processing query file: {args.query_json}")
    print(f"Query name: {query_json_name}")

    if DATE == "all":
        video_list = []
        video_dir = os.path.join(args.video_dir, NAME)
        for day in os.listdir(video_dir):
            day_path = os.path.join(video_dir, day)
            if os.path.isdir(day_path):
                video_list.extend(
                    [os.path.join(day_path, v) for v in os.listdir(day_path)]
                )
    else:
        video_dir = os.path.join(args.video_dir, NAME)
        day_path = os.path.join(video_dir, DATE)
        video_list = [os.path.join(day_path, video) for video in os.listdir(day_path)]
    video_list.sort()

    model_class = import_model(config["model"]["name"])
    # Initialize RAG agent
    rag_agent = RagAgent(
        model=model_class(**config["model"]["params"]),
        database_t=Chroma(name=DB_NAME),  # preset database path
        video_base_dir=video_dir,
        name=NAME,
    )

    if "create" in args.stage:
        print("create stage")
        caption_args = config["caption"]
        rag_agent.create_database_from_query(
            video_paths=video_list,
            query_json=query_data,
            caption_args=caption_args,
            human_query="Imagine you are the character in the video, describing from a first-person perspective what you saw, and everything that happened over time. Use I as the subject.",
            system_message="You are an egocentric agent. Take yourself as the main character of the video.",
            rag="rag_CLIP_t",
        )
        gen_event(DB_NAME, args.diary_dir)
    if "query" in args.stage:
        print("query stage")
        # Generate filename with current time and db_name
        output_dir = args.query_output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Add query_json_name to the output filename
        output_filename = f"{RUN_TIME}_{DB_NAME}_{query_json_name}_results.json"
        output_filepath = os.path.join(output_dir, output_filename)
        
        hour_event_file = os.path.join(args.diary_dir, f"{DB_NAME}_l2/merged_day_1to7.json")
        date_event_file = os.path.join(args.diary_dir, f"{DB_NAME}_l3/merged_day_1to7.json")
        
        # Check if diary files exist
        if not os.path.exists(hour_event_file) or not os.path.exists(date_event_file):
            raise FileNotFoundError(f"Diary files not found in {args.diary_dir}. Please check the diary_dir path.")
            
        with open(hour_event_file, "r") as f:
            event_data = json.load(f)
        with open(date_event_file, "r") as f:
            date_event_data = json.load(f)

        query_results = rag_agent.query_all(
            query_data=query_data,
            event_data=event_data,
            date_event_data=date_event_data,
        )
        # Save query results to the file
        with open(output_filepath, "w", encoding="utf-8") as outfile:
            json.dump(query_results, outfile, ensure_ascii=False, indent=4)

        print(f"Query results saved to {output_filepath}")

    if "answer" in args.stage:
        print("answer stage")
        output_dir = args.answer_output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Add query_json_name to the output filename
        output_filename = f"{RUN_TIME}_{DB_NAME}_{query_json_name}_results.json"
        output_filepath = os.path.join(output_dir, output_filename)

        # Load query results either from specified file or from previous stage
        if args.query_result_json:
            with open(args.query_result_json, "r", encoding="utf-8") as infile:
                query_results = json.load(infile)
            print(f"Loaded query results from {args.query_result_json}")
        elif "query_results" not in locals():
            raise ValueError(
                "No query results available. Either run 'query' stage or provide --query_result_json"
            )

        answers = []
        for query_result in tqdm(query_results, desc="Answering queries"):
            try:
                query_range = query_result["result"]
                question = query_result["question"]
                question_time = query_result["metadata"]["query_time"]
                formatted_question_time = (
                    f"{question_time['date']} {question_time['time']}"
                )
                formatted_options = query_result["formatted_options"]

                evidence_cards, caption_results = rag_agent.extract_evidence(
                    question, query_range, modality="text"
                )

                ans, answer_option = rag_agent.get_answer(
                    question,
                    formatted_question_time,
                    formatted_options,
                    evidence_cards,
                    caption_results=caption_results,
                )

                answers.append(
                    {
                        "metadata": query_result["metadata"],
                        "model_answer": ans,
                        "model_option": answer_option,
                        "evidence_card": evidence_cards,
                    }
                )
            except Exception as e:
                print(f"Error processing query: {e}")
                continue

        try:
            acc = rag_agent.calculate_accuracy(answers, save_to_file=output_filepath)
        except Exception as e:
            print(f"Error calculating accuracy: {e}")
            acc = 0
        print("Accuracy is:", acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video data for RAG agent.")

    parser.add_argument(
        "--date",
        type=str,
        required=False,
        default="all",
        help="Date of the video directory",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        default="A1_JAKE",
        help="Name of the video directory",
    )
    parser.add_argument(
        "--db_name",
        type=str,
        required=True,
        default="DEFAULT_DB_NAME",
        help="Name of the database",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        default="Egolife",
        help="video directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/gpt_4o.yaml",
        help="Path to model config file",
    )
    parser.add_argument(
        "--stage",  # options: 'create', 'query', 'answer'
        nargs="+",
        default=["create", "query", "answer"],
        required=False,
        help="Stages to run: create database or query or answer",
    )

    parser.add_argument(
        "--query_json",
        type=str,
        default="data/JAKE_ALL.json",
        help="Path to the query JSON file",
    )

    parser.add_argument(
        "--query_result_json",
        type=str,
        default=None,
        help="Path to existing query results JSON file",
    )

    parser.add_argument(
        "--query_output_dir",
        type=str,
        default="query_results",
        help="Directory to save query results",
    )

    parser.add_argument(
        "--answer_output_dir",
        type=str,
        default="answer_results",
        help="Directory to save answer results",
    )

    parser.add_argument(
        "--diary_dir",
        type=str,
        default="events_diary",
        help="Directory containing diary files",
    )

    args = parser.parse_args()

    main(args)
