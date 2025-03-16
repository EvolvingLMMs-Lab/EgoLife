import json
import os
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional
import random
import numpy as np
import requests
from egorag.database.Chroma import Chroma
from egorag.models import BaseQueryModel
from egorag.utils.util import *
from egorag.utils.agent_fun import *
from egorag.utils.prompts import *
from tqdm import tqdm


import time
from typing import Optional

import requests




class RagAgent(ABC):
    def __init__(
        self,
        model: Optional[BaseQueryModel] = None,
        database_t: Optional[Chroma] = None,
        database_i: Optional[Chroma] = None,
        name: str = "NULL",
        video_base_dir: str = "data/videos",
    ):
        super().__init__()
        self.model = model
        self.database_t = database_t
        self.database_i = database_i
        self.name = name
        self.video_base_dir = video_base_dir

    def calculate_accuracy(self, answers, save_to_file=None):
        total_count = 0
        correct_count = 0
        results = []

        for answer in answers:
            correct_answer = answer["metadata"]["answer"]
            model_answer = answer["model_option"]

            total_count += 1
            if model_answer is None or not model_answer.strip():
                is_correct = False
            else:
                is_correct = model_answer.strip() == correct_answer.strip()

            correct_count += is_correct

            answer["is_correct"] = is_correct
            results.append(answer)
        accuracy = correct_count / total_count if total_count > 0 else 0

        if save_to_file:
            with open(save_to_file, "w", encoding="utf-8") as f:
                json.dump(
                    {"accuracy": accuracy, "results": results},
                    f,
                    ensure_ascii=False,
                    indent=4,
                )

        return accuracy

    def get_video_mapping(self, sorted_video_times, all_times):
        video_path_mapping = {}
        for day in sorted_video_times.keys():
            date = day
            base_dir = self.video_base_dir
            video_time_list = sorted_video_times[date]
            all_time_list = all_times[date]
            for i in range(len(all_time_list) - 1):
                start_time = all_time_list[i]
                end_time = all_time_list[i + 1]

                for j in range(len(video_time_list) - 1):
                    if video_time_list[j] <= start_time < video_time_list[j + 1]:
                        video_path = os.path.join(
                            base_dir,
                            f"{date}",
                            f"{date}_{self.name}_{video_time_list[j]}.mp4",
                        )
                        break
                else:
                    video_path = None

                video_path_mapping[f"{date}-{start_time}-{end_time}"] = video_path
        return video_path_mapping

    def add_to_db_t(self, id, sample, caption_args, human_query, system_message=None):
        time_start = time.time()
        start_time = sample["start_time"]
        end_time = sample["end_time"]
        video_path = sample["video_path"]
        date = sample["int_date"]
        video_start_time = sample["video_start_time"]

        try:
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("Inference took too long")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(180)

            caption = self.model.inference_video(
                video_path, video_start_time, start_time, end_time, human_query
            )
            signal.alarm(0)
            print(caption)
        except TimeoutError as te:
            print(f"Timeout error in captioning: {te}")
            return None
        except Exception as e:
            print(f"Error in captioning: {e}")
            return None

        if caption is None or caption == "" or caption == "Error in captioning":
            return None

        sentences = [
            sentence.strip() + "."
            for sentence in caption.split(".")
            if sentence.strip()
        ]
        embeddings = self.database_t.embedding_function(sentences)
        new_ids = [f"{id}_{i}" for i in range(len(sentences))]
        add_element = dict(
            ids=new_ids,
            documents=sentences,
            embeddings=embeddings,
            metadatas=[
                {
                    "start_time": int(start_time),
                    "end_time": int(end_time),
                    "video_path": video_path,
                    "date": date,
                }
            ]
            * len(sentences),
        )

        collection = self.database_t.collection
        collection.add(**add_element)

        print(f"Database_t: Segment {id} processed and added to the collection.")
        print(
            f"Current collection size: {len(collection.get(include=['documents'])['ids'])}"
        )
        print(f"Time cost: {time.time()-time_start:.2f}s")

        return [
            {"id": new_id, "caption": sentence}
            for new_id, sentence in zip(new_ids, sentences)
        ]


    def create_database_from_query(
        self,
        video_paths,
        query_json,
        human_query,
        caption_args,
        system_message=None,
        rag="rag_CLIP_t",
    ):
        video_time = [
            {
                "date": re.search(r"DAY\d", path).group(0),
                "time": re.search(r"_(\d{8})\.mp4", path).group(1),
            }
            for path in video_paths
            if re.search(r"DAY\d", path) and re.search(r"_(\d{8})\.mp4", path)
        ]

        sorted_video_time = sorted(video_time, key=lambda x: (x["date"], x["time"]))
        query_time = []
        seen_times = set()
        for data in query_json:
            date = data["query_time"]["date"]
            time = data["query_time"]["time"]
            if (date, time) not in seen_times:
                query_time.append({"date": date, "time": time})
                seen_times.add((date, time))
        sorted_query_time = sorted(query_time, key=lambda x: (x["date"], x["time"]))

        all_times = sorted(
            sorted_video_time + sorted_query_time, key=lambda x: (x["date"], x["time"])
        )

        sorted_video_time = transform_timedict(sorted_video_time)
        all_times = transform_timedict(all_times)
        video_mapping = self.get_video_mapping(sorted_video_time, all_times)
        samples = parse_video_map(video_mapping)

        if rag in ["rag_CLIP_t"]:
            ids = self.database_t.collection.get()["ids"]
            existing_video_ids = set(id.rsplit("_", 1)[0] for id in ids)
            for id, sample in tqdm(
                enumerate(samples), total=len(samples), desc="Generating caption"
            ):
                video_id = sample["video_id"]
                if video_id in existing_video_ids:
                    print(f"Already in database: {video_id}, skip.")
                    continue

                self.add_to_db_t(
                    video_id, sample, caption_args, human_query, system_message
                )

    def create_database_from_json(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"Loaded {len(data)} entries from {json_path}")

        for idx, entry in tqdm(
            enumerate(data), total=len(data), desc="Processing entries"
        ):
            try:
                date_match = re.search(r"DAY(\d+)", entry["date"])
                int_date = int(date_match.group(1)) if date_match else 0
                date = entry["date"]
                video_name = f'{date}_{self.name}_{entry["start_time"]}.mp4'
                video_path = entry.get(
                    "video_path",
                    os.path.join(self.video_base_dir, self.name, date, video_name),
                )

                sentences = [
                    sentence.strip()
                    for sentence in re.split(r"(?<=[.!?])\s*", entry["text"])
                    if sentence.strip()
                ]
                entry_id = f"{entry['date']}-{entry['start_time']}-{entry['end_time']}"
                new_ids = [f"{entry_id}_{i}" for i in range(len(sentences))]

                existing_docs = self.database_t.collection.get(
                    ids=new_ids, include=["documents"]
                )
                if existing_docs["ids"] and any(existing_docs["documents"]):
                    print(f"IDs {entry_id} already exist with content, skipping.")
                    continue

                metadata = {
                    "start_time": int(entry["start_time"]),
                    "end_time": int(entry["end_time"]),
                    "date": int_date,
                    "video_path": video_path,
                }

                embeddings = self.database_t.embedding_function(sentences)

                add_element = {
                    "ids": new_ids,
                    "documents": sentences,
                    "metadatas": [metadata] * len(sentences),
                    "embeddings": embeddings,
                }

                self.database_t.collection.add(**add_element)
                if idx % 100 == 0:
                    print(
                        f"Added {idx} entries. Current collection size: {len(self.database_t.collection.get(include=['documents'])['ids'])}"
                    )
            except Exception as e:
                print(f"Error processing entry {idx}: {e}")
                continue

        print("Database creation completed!")
        print(
            f"Final collection size: {len(self.database_t.collection.get(include=['documents'])['ids'])}"
        )

    def generate_event_data(self, q_date, q_start_time, q_end_time):
        docs_to_use = self.database_t.collection.get(
            where={
                "$and": [
                    {"date": {"$eq": q_date}},
                    {"start_time": {"$gte": q_start_time}},
                    {"end_time": {"$lte": q_end_time}},
                ]
            },
            include=["documents"],
        )

        context = docs_to_use["documents"]

        response = call_gpt4(
            system_message=gen_event_prompt,
            prompt=f"all descriptions: {context}",
        )

        if response:
            return response
        else:
            return ""

    def filter_event_data(self, date, time, hour_docs, date_docs):
        date = int(date[-1])
        time = int(time)
        date_data = [entry for entry in date_docs if entry["date"] < date]
        hour_data = [
            entry
            for entry in hour_docs
            if entry["date"] == date and entry["end_time"] <= time
        ]
        if hour_data == []:
            last_event = {}
            last_event["generated_text"] = self.generate_event_data(
                date, hour_docs[0]["start_time"], time
            )
            last_event["date"] = date
            last_event["start_time"] = hour_docs[0]["start_time"]
            last_event["end_time"] = time
            hour_data.append(last_event)

        elif hour_data[-1]["end_time"] < time:
            last_event = {}
            last_event["generated_text"] = self.generate_event_data(
                date, hour_data[-1]["end_time"], time
            )
            last_event["date"] = date
            last_event["start_time"] = hour_data[-1]["end_time"]
            last_event["end_time"] = time
            hour_data.append(last_event)

        all_date_data = date_data + hour_data
        return all_date_data, hour_data

    def process_query_results(self, raw_results):
        """
        Process raw query results into a list of individual results with specific fields.

        Args:
            raw_results (dict): Raw results dictionary from the database query

        Returns:
            list: List of dictionaries containing processed results
        """
        processed_results = []

        if not raw_results["ids"] or not raw_results["ids"][0]:
            return processed_results

        ids = raw_results["ids"][0]
        documents = raw_results["documents"][0]
        metadatas = raw_results["metadatas"][0]
        distances = raw_results["distances"][0]

        combined_results = [
            (
                ids[i],
                documents[i],
                metadatas[i]["date"],
                metadatas[i]["end_time"],
                distances[i],
            )
            for i in range(len(ids))
        ]

        combined_results.sort(key=lambda x: (x[2], x[3]))
        for id, document, date, end_time, distance in combined_results:
            results = self.database_t.get_caption(id=id, n_result=1)
            expand_documents = results["documents"]
            processed_result = {
                "id": id,
                "document": expand_documents,
                "date": date,
                "end_time": end_time,
                "distance": distance,
            }
            processed_results.append(processed_result)

        return processed_results

    def get_range(self, question, date, time, hour_docs, date_docs):
        int_date = int(date[-1])
        all_date, hour = self.filter_event_data(date, time, hour_docs, date_docs)


        prompt_l1 = f"all json {all_date}"
        response_l1 = call_gpt4(
            prompt_l1, system_message=get_day_prompt.format(question=question, date=date, time=time), temperature=0.8
        )
        match = re.match(r"\[(\d+)\]", response_l1.strip())
        if match:
            target_date = int(match.group(1))
        else:
            target_date = int_date
        if target_date == int_date:
            hours_data = hour
        else:
            hours_data = [entry for entry in hour_docs if entry["date"] == target_date]

        prompt_l2 = f"all description: {hours_data}"
        response_l2 = call_gpt4(
            prompt_l2, system_message=get_hour_prompt.format(question=question, date=date, time=time), temperature=0.8
        )
        start_time_match = re.search(r"\[(\d{8})\]", response_l2.split("\n")[0])
        end_time_match = re.search(r"\[(\d{8})\]", response_l2.split("\n")[1])

        start_time = (
            int(start_time_match.group(1))
            if start_time_match
            else hours_data[0]["start_time"]
        )
        end_time = (
            int(end_time_match.group(1))
            if end_time_match
            else hours_data[-1]["end_time"]
        )
        return target_date, start_time, end_time

    def query(self, query_dict, event_data, date_docs):
        question, query, formatted_options, time, date = self.parse_query(query_dict)
        if "last time" in question.lower() or "the last" in question.lower():

            prompt = f"Question: {question} \nOptions: \n{formatted_options}\nQuestion Time: {date} {time}\nOutput:"

            keyword_raw = call_gpt4(
                prompt=prompt, system_message=query_prompt, temperature=0.1
            )
            # Extract keywords and time range using regex
            keyword_pattern = r"Keywords:\s*\[([^\]]+)\]"
            time_range_pattern = r"Search Time Range:\s*\[([^\]]+)\]"

            # Extract keywords
            keyword_match = re.search(keyword_pattern, keyword_raw)
            keywords = keyword_match.group(1).strip() if keyword_match else question
            # Extract time range
            time_range_match = re.search(time_range_pattern, keyword_raw)
            if time_range_match:
                search_time = time_range_match.group(1).strip()
                # Split into day and time if the format is "DAYX XXXXXXXX"
                search_time_parts = search_time.split()
                search_date = (
                    int(search_time_parts[0].replace("DAY", ""))
                    if len(search_time_parts) > 1
                    else int(date[-1])
                )
                search_timestamp = (
                    int(search_time_parts[1])
                    if len(search_time_parts) > 1
                    else int(search_time)
                )
            else:
                search_date = int(date[-1])
                search_timestamp = int(time)
            raw_results = self.database_t.custom_query(
                query_texts=[keywords],
                n_results=50,
                where={
                    "$or": [
                        {"date": {"$lt": search_date}},
                        {
                            "$and": [
                                {"end_time": {"$lte": search_timestamp}},
                                {"date": {"$eq": search_date}},
                            ]
                        },
                    ]
                },
                filter_first=False,
            )
            all_query_results = self.process_query_results(raw_results)
            filter_id = get_ids(question=question, results=all_query_results)
            results = self.database_t.get_caption(id=filter_id, n_result=1)
            best_result = results["item"]
            query_result = [
                {
                    "query_range": None,
                    "docs": best_result,
                    "start_time": None,
                    "end_time": None,
                    "date": None,
                    "extract_keywords": keywords,
                    "filter_id": filter_id,
                    "search_range": search_time,
                }
            ]

        else:
            date, start_time, end_time = self.get_range(
                question, date, time, event_data, date_docs
            )
            docs_in_range = self.database_t.custom_query(
                query_texts=[query],
                n_results=3,
                where={
                    "$and": [
                        {"start_time": {"$gte": start_time}},
                        {"end_time": {"$lte": end_time}},
                        {"date": {"$eq": date}},
                    ]
                },
                filter_first=True,
            )
            query_result = [
                {
                    "query_range": [date, start_time, end_time],
                    "docs": docs_in_range,
                    "start_time": start_time,
                    "end_time": end_time,
                    "date": date,
                }
            ]

        return query_result, question, formatted_options

    def single_query(self, query_dict, event_data, date_event_data):
        query_result, question, formatted_options = self.query(
            query_dict, event_data, date_event_data
        )
        return query_result, question, formatted_options

    def query_all(self, query_data, event_data, date_event_data):
        all_query_results = []

        for query_dict in tqdm(query_data, desc="Processing queries"):
            try:
                query_result, question, formatted_options = self.single_query(
                    query_dict, event_data, date_event_data
                )

                all_query_results.append(
                    {
                        "metadata": query_dict,
                        "result": query_result,
                        "question": question,
                        "formatted_options": formatted_options,
                    }
                )
            except Exception as e:
                print(f"error {e}")
                continue

        return all_query_results

    def parse_query(self, data_dict):
        """
        Generate a question prompt for a model based on the given dictionary.

        Args:
            data_dict (dict): A dictionary containing date, time, query, options, and answer.

        Returns:
            str: A formatted prompt for the model.
        """
        # Extract values from the dictionary
        query_time = data_dict.get("query_time", None)

        date = query_time.get("date", "DAY6")
        time = query_time.get("time", "18000000")
        question = data_dict.get("question", "UNKNOWN_QUERY")
        query = data_dict.get("keywords", "UNKNOWN_QUERY")
        # Extract choices from the dictionary
        options = []
        for letter in ["a", "b", "c", "d"]:
            choice_key = f"choice_{letter}"
            if choice_key in data_dict:
                options.append(f"{letter.upper()}. {data_dict[choice_key]}")

        # Format the options
        formatted_options = "\n".join(options)

        return question, query, formatted_options, time, date

    # CoT filter get evidence
    def extract_evidence(self, question, query_range, modality="video"):
        def parse_evidence_output(response):
            """
            Parse the LLM output and return a dictionary with the appropriate format.

            Parameters:
            - response (str): The response from the LLM.

            Returns:
            - dict: A dictionary with the extracted status and information (if applicable).
            """
            if "I can't provide evidence." in response:
                return {"status": False}

            match = re.search(r"I can provide evidence\. Evidence: (.+)", response)
            if match:
                return {"status": True, "information": match.group(1).strip()}

            return {"status": False}

        doc_ids = []
        evidence = []
        caption_results = {}
        single_range = query_range[0]["docs"]
        for index, id in enumerate(single_range["ids"]):
            video_start_time = (
                single_range["metadatas"][index]["video_path"]
                .split(".")[0]
                .split("_")[-1]
            )
            start_time = single_range["metadatas"][index]["start_time"]
            end_time = single_range["metadatas"][index]["end_time"]
            video_path = single_range["metadatas"][index]["video_path"]
            video_date = single_range["metadatas"][index]["date"]

            all_result = self.database_t.get_caption(id=id, n_result=10)

            full_caption = " ".join(all_result["documents"])
            video_caption = f"""
            Video Time: It is DAY{video_date}, {start_time} to {end_time}.
            Video Conetent: {full_caption}
            """
            caption_key = f"video{index+1}"
            caption_results[caption_key] = video_caption
            doc_ids.append(single_range["ids"][index])
            if modality == "video":
                video_prompt = video_evidence_prompt.format(question=question)

                response = self.model.inference_video(
                    video_path,
                    int(video_start_time),
                    start_time,
                    end_time,
                    video_prompt,
                )
                evidence.append(parse_evidence_output(response))
            elif modality == "text":
                text_prompt = text_evidence_prompt.format(video_caption=video_caption,question=question)

                response = call_gpt4(text_prompt)
                evidence.append(parse_evidence_output(response))
            else:
                raise ValueError(f"Invalid modality: {modality}")

        evidence_cards = []
        for index, id in enumerate(doc_ids):
            if evidence[index]["status"]:
                evidence_card = {"time": id}

                if evidence[index]["status"]:
                    evidence_card["evidence_info"] = evidence[index]["information"]
                else:
                    evidence_card["evidence_info"] = "No evidence information."

                evidence_cards.append(evidence_card)
        return evidence_cards, caption_results


    # use filtered and collected evidence_cards to get answer from language model
    def get_answer(
        self, question, question_time, formatted_options, evidence_cards, caption_results
    ):
        combined_string = "\n\n".join(
            f"time: {d.get('time', 'N/A')}\n"
            f"evidence_info: {d.get('evidence_info', 'No evidence info')}\n"
            for d in evidence_cards
        )

        prompt_with_evidence = with_evidence.format(combined_string=combined_string, question=question, question_time=question_time, formatted_options=formatted_options)

        prompt_without_evidence = without_evidence.format(caption_results=caption_results, question=question, question_time=question_time, formatted_options=formatted_options)
        if evidence_cards == []:
            ans = call_gpt4(
                prompt=prompt_without_evidence,
                system_message="You are an AI assistant that answers questions based on video content.",
                temperature=0.1,
            )
        else:
            ans = call_gpt4(
                prompt=prompt_with_evidence,
                system_message="You are an AI assistant that answers questions based on video content.",
                temperature=0.1,
            )
        if ans == "error":
            choices = ["A", "B", "C", "D"]
            ans = "answer error, random choice"
            answer_option = random.choice(choices)
        else:
            answer_option = extract_single_option_answer(ans)

        return ans, answer_option
