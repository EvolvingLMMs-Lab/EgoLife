import argparse
import json
import os
from collections import defaultdict

import requests
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.pipeline.transport import RequestsTransport
from egorag.database.Chroma import Chroma
from egorag.utils.util import call_gpt4,call_deepseek


system_message_min = f"""As an **Event Summary Documentation Specialist**, your role is to systematically structure and summarize event information, ensuring that all key actions of major characters are captured while maintaining clear event logic and completeness. Your focus is on concise and factual summarization rather than detailed transcription.

    #### **Specific Requirements**

    1. **Structure the Events Clearly**
    - **Merge related events:** Consolidate similar content into major events and arrange them in chronological order to ensure a smooth logical flow.
    - **Logical segmentation:** Events can be grouped based on location, task, or theme. Each event should have a clear starting point, progression, and key turning points without any jumps or fragmentation in the information.

    2. **Retain Key Information**
    - The primary character's (**"I"**) decisions and actions must be fully presented, including all critical first-person activities. Transitions between different parts, such as moving between floors or starting/ending a task, should be seamless.
    - Any discussions, decisions, and task execution involving the primary character and other key individuals that impact the main storyline must be reflected. This includes recording, planning, and confirming matters, but in a concise manner.
    - The **purpose and method** of key actions must be recorded, such as "ordering takeout using a phone" or "documenting a plan on a whiteboard."

    3. **Concise Expression, Remove Redundancies**
    - Keep the facts clear, avoiding descriptions of atmosphere, emotions, or abstract content.
    - Remove trivial conversations and extract only the core topics and conclusions of discussions. If a discussion is lengthy, summarize it into **task arrangements, decision points, and specific execution details**.

    4. **Strictly Adhere to Facts, No Assumptions**
    - Do not make assumptions or add interpretations—strictly organize content based on available information, ensuring accuracy. Every summarized point must have a basis in the original information, with no unnecessary additions.
    - Maintain the correct **chronological order** of events. The sequence of developments must strictly follow their actual occurrence without any inconsistencies.

    #### **Output Format**
    Each paragraph should represent **one major event**, structured in a **summary-detail-summary** format. Do not report the word count in the output.
    Example output of one event: 
    **[write down the summary of the event, no index]** One sentence summarizing the overall situation + Description of my key actions and other relevant characters' critical actions + Conclusion/recap to ensure logical clarity.
    """

system_message_hour = f"""As an **Event Summary Documentation Specialist**, your role is to systematically structure and summarize event information, ensuring that all key actions of major characters are captured while maintaining clear event logic and completeness. Your focus is on concise and factual summarization rather than detailed transcription.

    #### **Specific Requirements**

    1. **Structure the Events Clearly**
    - **Merge related events:** Consolidate similar content into major events and arrange them in chronological order to ensure a smooth logical flow.
    - **Logical segmentation:** Events can be grouped based on location, task, or theme. Each event should have a clear starting point, progression, and key turning points without any jumps or fragmentation in the information.

    2. **Retain Key Information**
    - The primary character's (**"I"**) decisions and actions must be fully presented, including all critical first-person activities. Transitions between different parts, such as moving between floors or starting/ending a task, should be seamless.
    - Any discussions, decisions, and task execution involving the primary character and other key individuals that impact the main storyline must be reflected. This includes recording, planning, and confirming matters, but in a concise manner.
    - The **purpose and method** of key actions must be recorded, such as "ordering takeout using a phone" or "documenting a plan on a whiteboard."

    3. **Concise Expression, Remove Redundancies**
    - Keep the facts clear, avoiding descriptions of atmosphere, emotions, or abstract content.
    - Remove trivial conversations and extract only the core topics and conclusions of discussions. If a discussion is lengthy, summarize it into **task arrangements, decision points, and specific execution details**.

    4. **Strictly Adhere to Facts, No Assumptions**
    - Do not make assumptions or add interpretations—strictly organize content based on available information, ensuring accuracy. Every summarized point must have a basis in the original information, with no unnecessary additions.
    - Maintain the correct **chronological order** of events. The sequence of developments must strictly follow their actual occurrence without any inconsistencies.

    #### **Output Format**
    Each paragraph should represent **one major event**, structured in a **summary-detail-summary** format. Strictly output below 500 words in total. Do not report the word count in the output.
    Example output of one event: 
    **[write down the summary of the event, no index]** One sentence summarizing the overall situation + Description of my key actions and other relevant characters' critical actions + Conclusion/recap to ensure logical clarity.
    """

system_message_day = f"""As an **Event Summary Documentation Specialist**, your role is to systematically structure and summarize event information of a whole day, ensuring that all key actions of major characters are captured while maintaining clear event logic and completeness. Your focus is on concise and factual summarization rather than detailed transcription.

    #### **Specific Requirements**

    1. **Structure the Events Clearly**
    - **Merge related events:** Consolidate similar content into major events and arrange them in chronological order to ensure a smooth logical flow.
    - **Logical segmentation:** Events can be grouped based on location, task, or theme. Each event should have a clear starting point, progression, and key turning points without any jumps or fragmentation in the information.

    2. **Retain Key Information**
    - The primary character's (**"I"**) decisions and actions must be fully presented, including all critical first-person activities. Transitions between different parts, such as moving between floors or starting/ending a task, should be seamless.
    - Any discussions, decisions, and task execution involving the primary character and other key individuals that impact the main storyline must be reflected. This includes recording, planning, and confirming matters, but in a concise manner.
    - The **purpose and method** of key actions must be recorded, such as "ordering takeout using a phone" or "documenting a plan on a whiteboard."

    3. **Concise Expression, Remove Redundancies**
    - Keep the facts clear, avoiding descriptions of atmosphere, emotions, or abstract content.
    - Remove trivial conversations and extract only the core topics and conclusions of discussions. If a discussion is lengthy, summarize it into **task arrangements, decision points, and specific execution details**.

    4. **Strictly Adhere to Facts, No Assumptions**
    - Do not make assumptions or add interpretations—strictly organize content based on available information, ensuring accuracy. Every summarized point must have a basis in the original information, with no unnecessary additions.
    - Maintain the correct **chronological order** of events. The sequence of developments must strictly follow their actual occurrence without any inconsistencies.

    #### **Output Format**
    Each paragraph should represent **one major event**, structured in a **summary-detail-summary** format. The output should be around 1500 words in total. Do not report the word count in the output.
    Example output of one event: 
    **[write down the summary of the event, no index]** One sentence summarizing the overall situation of the day + Description of my key actions and other relevant characters' critical actions + Conclusion/recap to ensure logical clarity.
    """


def time_to_minutes(time):
    hours = time // 1000000
    minutes = (time % 1000000) // 10000
    seconds = (time % 10000) // 100
    return hours * 60 + minutes + seconds / 60


def minutes_to_time(minutes):
    hours = int(minutes // 60)
    minutes = int(minutes % 60)
    seconds = int((minutes - int(minutes)) * 60)
    return hours * 1000000 + minutes * 10000 + seconds * 100


def process_day(docs, date):
    results = []
    batch = []
    batch_start_time = None
    current_window_start = None
    window_index = 0

    for idx, doc in enumerate(docs):
        start_time = time_to_minutes(doc["Metadata"]["start_time"])
        end_time = time_to_minutes(doc["Metadata"]["end_time"])

        if current_window_start is None:
            current_window_start = start_time
            current_window_end = current_window_start + 10

        if start_time < current_window_end:
            batch.append(doc)
        else:
            if batch:
                context = []
                for item in batch:
                    context.append(item["Content"])
                try:
                    gpt_res = call_gpt4(
                        prompt=f"All descriptions: {context}",
                        system_message=system_message_min,
                        max_tokens=2048,
                        temperature=0.7,
                        top_p=0.95
                    )
                except Exception as e:
                    print(f"Error processing batch {window_index}: {e}")
                    gpt_res = call_deepseek(
                        prompt=f"All descriptions: {context}",
                        system_message=system_message_min
                    )
                print(gpt_res)
                if gpt_res:
                    try:
                        res = gpt_res
                        results.append(
                            {
                                "generated_text": res,
                                "date": date,
                                "start_time": minutes_to_time(current_window_start),
                                "end_time": minutes_to_time(current_window_end),
                            }
                        )
                        print(f"Batch {window_index} processed successfully.")
                    except (KeyError, IndexError, json.JSONDecodeError) as e:
                        print(
                            f"Error extracting content from response in batch {window_index}: {e}"
                        )
                else:
                    print(f"Error processing batch {window_index}")

            current_window_start = (start_time // 10) * 10
            current_window_end = current_window_start + 10
            window_index += 1
            batch = [doc]

    if batch:
        context = []
        for item in batch:
            context.append(item["Content"])
        
        gpt_res = call_gpt4(
            prompt=f"All descriptions: {context}",
            system_message=system_message_min,
            max_tokens=2048,
            temperature=0.7,
            top_p=0.95
        )
        
        if gpt_res:
            try:
                res = gpt_res
                results.append(
                    {
                        "generated_text": res,
                        "date": date,
                        "start_time": minutes_to_time(current_window_start),
                        "end_time": minutes_to_time(current_window_end),
                    }
                )
                print(f"Batch {window_index} processed successfully.")
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                print(
                    f"Error extracting content from response in batch {window_index}: {e}"
                )
        else:
            print(f"Error processing final batch.")
    print(results)
    return results


def get_event_min(db_name,diary_dir):
    output_dir = f"./{diary_dir}/{db_name}_l1"
    if os.path.exists(output_dir):
        print(
            f"Directory {output_dir} already exists. Checking for incomplete files..."
        )
    else:
        os.makedirs(output_dir, exist_ok=True)

    database = Chroma(name=db_name)
    docs = database.get_doc()

    docs_by_date = defaultdict(list)
    for doc in docs:
        date = doc["Metadata"].get("date")
        docs_by_date[date].append(doc)
    for date in range(1, 8):
        output_file = f"./{diary_dir}/{db_name}_l1/l1_day{date}.json"

        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    json.load(f)
                print(f"File for Date {date} already exists and is valid. Skipping...")
                continue
            except json.JSONDecodeError:
                print(f"Existing file for Date {date} is corrupt. Regenerating...")

        if date in docs_by_date:
            results = process_day(docs_by_date[date], date)
            temp_file = f'{output_file.split(".")[0]}_tmp.json'
            try:
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                os.replace(temp_file, output_file)
                print(f"Results for Date {date} saved successfully.")
            except Exception as e:
                print(f"Error saving results for Date {date}: {e}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        else:
            print(f"No data for Date {date}.")

    json_files = [
        f for f in os.listdir(f"./{diary_dir}/{db_name}_l1") if f.endswith(".json")
    ]
    merged_items = []
    for file in json_files:
        file_path = os.path.join(f"./{diary_dir}/{db_name}_l1", file)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            merged_items.extend(data)

    with open(
        f"./{diary_dir}/{db_name}_l1/merged_day_1to7.json", "w", encoding="utf-8"
    ) as f:
        json.dump(merged_items, f, ensure_ascii=False, indent=4)

    print("JSON combined successfully")


def process_json_batches(input_file, output_file, date):
    with open(input_file, "r") as f:
        data = json.load(f)

    results = []
    batch = []
    current_batch_start_time = 0
    batch_index = 1

    for item in data:
        start_time_str = str(item["start_time"])
        if len(str(item["start_time"])) < 8:
            start_hour = int(start_time_str[:1])
            start_minute = int(start_time_str[1:3])
        else:
            start_hour = int(start_time_str[:2])
            start_minute = int(start_time_str[2:4])

        item_start_time = start_hour * 3600 + start_minute * 60

        if not batch:
            current_batch_start_time = item_start_time
            batch.append(item)
        else:
            current_batch_end_time = current_batch_start_time + 3600
            if item_start_time < current_batch_end_time:
                batch.append(item)
            else:
                context = "\n".join(item["generated_text"] for item in batch)
                gpt_res = call_deepseek(
                    prompt=f"All descriptions: {context}. Strictly output below 500 words in total.",
                    system_message=system_message_hour,
                )
                if gpt_res:
                    try:
                        res = gpt_res
                        results.append(
                            {
                                "generated_text": res,
                                "date": date,
                                "start_time": batch[0]["start_time"],
                                "end_time": batch[-1]["end_time"],
                            }
                        )
                        print(f"Batch {batch_index} processed successfully.")
                    except (KeyError, IndexError, json.JSONDecodeError) as e:
                        print(
                            f"Error extracting content from response in batch {batch_index}: {e}"
                        )
                    batch_index += 1

                batch = [item]
                current_batch_start_time = item_start_time

    if batch:
        context = "\n".join(item["generated_text"] for item in batch)
        gpt_res = call_deepseek(
            prompt=f"All descriptions: {context}. Strictly output below 500 words in total.",
            system_message=system_message_hour,
        )
        if gpt_res:
            try:
                res = gpt_res
                results.append(
                    {
                        "generated_text": res,
                        "date": date,
                        "start_time": batch[0]["start_time"],
                        "end_time": batch[-1]["end_time"],
                    }
                )
                print(f"Batch {batch_index} processed successfully.")
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                print(
                    f"Error extracting content from response in batch {batch_index}: {e}"
                )

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)


def get_event_hour(db_name,diary_dir):
    output_dir = f"./{diary_dir}/{db_name}_l2"
    if os.path.exists(output_dir):
        print(
            f"Directory {output_dir} already exists. Checking for incomplete files..."
        )
    else:
        os.makedirs(output_dir, exist_ok=True)

    for i in range(1, 8):
        input_file = f"./{diary_dir}/{db_name}_l1/l1_day{i}.json"
        output_file = f"./{diary_dir}/{db_name}_l2/l2_day{i}.json"

        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    json.load(f)
                print(f"L2 file for Day {i} already exists and is valid. Skipping...")
                continue
            except json.JSONDecodeError:
                print(f"Existing L2 file for Day {i} is corrupt. Regenerating...")

        if not os.path.exists(input_file):
            print(f"Input file for Day {i} does not exist. Skipping...")
            continue

        try:
            process_json_batches(
                input_file, f'{output_file.split(".")[0]}_tmp.json', date=i
            )
            os.replace(f'{output_file.split(".")[0]}_tmp.json', output_file)
            print(f"L2 results for Day {i} saved successfully.")
        except Exception as e:
            print(f"Error processing Day {i}: {e}")
            if os.path.exists(f'{output_file.split(".")[0]}_tmp.json'):
                os.remove(f'{output_file.split(".")[0]}_tmp.json')
            with open("error_file.json", "w") as f:
                json.dump([i, input_file, output_file], f, indent=4)

    json_files = [
        f for f in os.listdir(f"./{diary_dir}/{db_name}_l2") if f.endswith(".json")
    ]
    merged_items = []
    for file in json_files:
        file_path = os.path.join(f"./{diary_dir}/{db_name}_l2", file)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            merged_items.extend(data)

    with open(
        f"./{diary_dir}/{db_name}_l2/merged_day_1to7.json", "w", encoding="utf-8"
    ) as f:
        json.dump(merged_items, f, ensure_ascii=False, indent=4)

    print("JSON combined successfully")


def get_event_day(db_name,diary_dir):
    output_dir = f"./{diary_dir}/{db_name}_l3"
    if os.path.exists(output_dir):
        print(
            f"Directory {output_dir} already exists. Checking for incomplete files..."
        )
    else:
        os.makedirs(output_dir, exist_ok=True)

    results = []
    for day in range(1, 8):
        input_file = f"./{diary_dir}/{db_name}_l2/l2_day{day}.json"
        output_file = f"./{diary_dir}/{db_name}_l3/l3_day{day}.json"

        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    json.load(f)
                print(f"L3 file for Day {day} already exists and is valid. Skipping...")
                continue
            except json.JSONDecodeError:
                print(f"Existing L3 file for Day {day} is corrupt. Regenerating...")

        try:
            with open(input_file, "r") as f:
                day_data = json.load(f)

            context = "\n".join(item["generated_text"] for item in day_data)
            gpt_res = call_gpt4(
                prompt=f"All descriptions: {context}. The output should be around 1500 words in total.",
                system_message=system_message_day,
                max_tokens=2048,
                temperature=0.7,
                top_p=0.95
            )
            if gpt_res:
                try:
                    res = gpt_res
                    day_summary = {
                        "generated_text": res,
                        "date": day,
                        "start_time": day_data[0]["start_time"],
                        "end_time": day_data[-1]["end_time"],
                    }

                    with open(f'{output_file.split(".")[0]}_tmp.json', "w") as f:
                        json.dump([day_summary], f, indent=4)
                    os.replace(f'{output_file.split(".")[0]}_tmp.json', output_file)

                    results.append(day_summary)
                    print(f"Day {day} processed successfully.")
                except (KeyError, IndexError, json.JSONDecodeError) as e:
                    print(f"Error extracting content from response for day {day}: {e}")
        except Exception as e:
            print(f"Error processing day {day}: {e}")
            if os.path.exists(f'{output_file.split(".")[0]}_tmp.json'):
                os.remove(f'{output_file.split(".")[0]}_tmp.json')
            # save the error day, start_time, and end_time to a file
            with open("error_file.json", "w") as f:
                json.dump(
                    [day, day_data[0]["start_time"], day_data[-1]["end_time"]],
                    f,
                    indent=4,
                )

    with open(
        f"./{diary_dir}/{db_name}_l3/merged_day_1to7.json", "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("Daily summaries generated and merged successfully")


# main
def gen_event(db_name, diary_dir):

    # Use the provided arguments in the functions
    get_event_min(db_name, diary_dir)
    get_event_hour(db_name, diary_dir)
    get_event_day(db_name, diary_dir)
