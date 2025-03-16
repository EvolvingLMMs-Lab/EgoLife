import os
from pathlib import Path
import re
import time
from typing import Optional
import yaml
import requests
import openai
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.pipeline.transport import RequestsTransport
from openai import OpenAI


def call_gpt4(
    prompt: str,
    system_message: str = "You are an effective first perspective assistant.",
    temperature=0.9,
    top_p=0.95,
    max_tokens=2200,
) -> Optional[str]:
    """
    Call GPT-4 API with given prompt and system message.
    Will automatically use Azure OpenAI if AZURE_GPT4_KEY and AZURE_GPT4_ENDPOINT are set,
    otherwise will fall back to standard OpenAI API using OPENAI_API_KEY.

    Args:
        prompt (str): The user prompt to send to GPT-4
        system_message (str): System message to set context for GPT-4
        temperature (float): Temperature parameter for response generation
        top_p (float): Top p parameter for response generation

    Returns:
        Optional[str]: The response content from GPT-4, or None if request fails after retries
    """
    azure_key = os.getenv("AZURE_API_KEY")
    azure_endpoint = os.getenv("AZURE_API_ENDPOINT")
    openai_key = os.getenv("OPENAI_API_KEY")

    # Use Azure if both Azure credentials are available
    if azure_key and azure_endpoint:
        headers = {
            "Content-Type": "application/json",
            "api-key": azure_key,
        }

        payload = {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": system_message}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        retries = 5
        for attempt in range(retries):
            try:
                response = requests.post(azure_endpoint, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"Attempt {attempt + 1} failed. Error: {e}")
                if attempt < retries - 1:
                    time.sleep(2)
                else:
                    print("All retry attempts failed.")
                    return "error"
    # Fall back to standard OpenAI API if OpenAI key is available
    elif openai_key:
        openai.api_key = openai_key
        
        retries = 5
        for attempt in range(retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=2200
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Attempt {attempt + 1} failed. Error: {e}")
                if attempt < retries - 1:
                    time.sleep(2)
                else:
                    print("All retry attempts failed.")
                    return "error"
    else:
        raise ValueError("Neither Azure OpenAI nor standard OpenAI credentials are properly configured")

def call_deepseek(
    prompt: str,
    system_message: str = "You are a helpful assistant.",
    max_tokens: int = 2048
) -> Optional[str]:
    """
    Call DeepSeek API with given prompt and system message.
    Will automatically use Azure DeepSeek if DEEPSEEK_KEY and DEEPSEEK_ENDPOINT are set,
    otherwise will fall back to standard DeepSeek API using DEEPSEEK_API_KEY.

    Args:
        prompt (str): The user prompt
        system_message (str): System message to set context
        max_tokens (int): Maximum number of tokens to generate

    Returns:
        Optional[str]: The response content, or None if request fails after retries
    """
    azure_key = os.getenv("AZURE_DEEPSEEK_KEY")
    azure_endpoint = os.getenv("AZURE_DEEPSEEK_ENDPOINT")
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

    # Use Azure if both Azure credentials are available
    if azure_key and azure_endpoint:
        model_name = os.getenv("DEPLOYMENT_NAME", "DeepSeek-R1")
        client = ChatCompletionsClient(
            endpoint=azure_endpoint,
            credential=AzureKeyCredential(azure_key),
            transport=RequestsTransport(read_timeout=500),
        )

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = client.complete(
                    messages=[
                        SystemMessage(content=system_message),
                        UserMessage(content=prompt),
                    ],
                    model=model_name,
                    max_tokens=max_tokens,
                )
                full_response = response["choices"][0]["message"]["content"]
                # Only split if the response contains </think> tag
                if "</think>" in full_response:
                    response = full_response.split("</think>")[1]
                    return response
                return full_response
            except Exception as e:
                retry_count += 1
                print(f"Error on attempt {retry_count}: {e}")
                if retry_count == max_retries:
                    print("Max retries reached. Giving up.")
                    return None

    # Fall back to standard DeepSeek API if API key is available
    elif deepseek_api_key:
        client = OpenAI(
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com"
        )

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    stream=False
                )
                full_response = response["choices"][0]["message"]["content"]
                if "</think>" in full_response:
                    response = full_response.split("</think>")[1]
                    return response
                return full_response
            except Exception as e:
                retry_count += 1
                print(f"Error on attempt {retry_count}: {e}")
                if retry_count == max_retries:
                    print("Max retries reached. Giving up.")
                    return None
    else:
        raise ValueError("Neither Azure DeepSeek nor standard DeepSeek credentials are properly configured")

def time_to_frame_idx(time_int: int, fps: int) -> int:
    """
    Convert time in HHMMSSFF format (integer or string) to frame index.
    :param time_int: Time in HHMMSSFF format, e.g., 10483000 (10:48:30.00) or "10483000".
    :param fps: Frames per second of the video.
    :return: Frame index corresponding to the given time.
    """
    # Ensure time_int is a string for slicing
    time_str = str(time_int).zfill(
        8
    )  # Pad with zeros if necessary to ensure it's 8 digits

    hours = int(time_str[:2])
    minutes = int(time_str[2:4])
    seconds = int(time_str[4:6])
    frames = int(time_str[6:8])

    total_seconds = hours * 3600 + minutes * 60 + seconds
    total_frames = total_seconds * fps + frames  # Convert to total frames

    return total_frames


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_video_paths(base_path, name):
    video_dir = Path(base_path) / name
    return [str(video_dir / video) for video in os.listdir(video_dir)]

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

def extract_single_option_answer(answer):
    """
    Extracts the single option (A, B, C, or D) from the given answer string.

    Args:
        answer (str): The answer string containing the selected option.

    Returns:
        str: The selected option (A, B, C, or D).
    """
    match = re.search(r"\b([A-D])\b", answer)
    if match:
        return match.group(1)
    return None