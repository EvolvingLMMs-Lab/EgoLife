import copy
import os
import re
import sys
import warnings

import numpy as np
import requests
import soundfile as sf
import torch
import torch.distributed as dist
import whisper
from decord import VideoReader, cpu
from egogpt.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_SPEECH_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    SPEECH_TOKEN_INDEX,
)
from egogpt.conversation import SeparatorStyle, conv_templates
from egogpt.mm_utils import get_model_name_from_path, process_images
from egogpt.model.builder import load_pretrained_model
from PIL import Image
from scipy.signal import resample


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def load_video(video_path=None, audio_path=None, max_frames_num=16, fps=1):
    if audio_path is not None:
        speech, sample_rate = sf.read(audio_path)
        if sample_rate != 16000:
            target_length = int(len(speech) * 16000 / sample_rate)
            speech = resample(speech, target_length)
        if speech.ndim > 1:
            speech = np.mean(speech, axis=1)
        speech = whisper.pad_or_trim(speech.astype(np.float32))
        speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0)
        speech_lengths = torch.LongTensor([speech.shape[0]])
    else:
        speech = torch.zeros(3000, 128)
        speech_lengths = torch.LongTensor([3000])

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    avg_fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]
    if max_frames_num > 0 and len(frame_idx) > max_frames_num:
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
    video = vr.get_batch(frame_idx).asnumpy()
    return video, speech, speech_lengths


def split_text(text, keywords):
    pattern = "(" + "|".join(map(re.escape, keywords)) + ")"
    parts = re.split(pattern, text)
    parts = [part for part in parts if part]
    return parts


def main():
    warnings.filterwarnings("ignore")
    setup(0, 1)

    pretrained = "checkpoints/EgoGPT-llavaov-7b-EgoIT-100k"
    device = "cuda"
    device_map = "cuda"

    tokenizer, model, max_length = load_pretrained_model(pretrained, device_map=device_map)
    model.eval()

    video_path = "data/train/A1_JAKE/DAY1/DAY1_A1_JAKE_11223000.mp4"
    audio_path = "audio/DAY1_A1_JAKE_11223000.mp3"

    conv_template = "qwen_1_5"
    question = "<image>\n<speech>\n\nPlease describe the video in detail."
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    video, speech, speech_lengths = load_video(video_path=video_path, audio_path=audio_path)
    speech = torch.stack([speech]).to(device).half()
    processor = model.get_vision_tower().image_processor
    processed_video = processor.preprocess(video, return_tensors="pt")["pixel_values"]
    image = [(processed_video, video[0].size, "video")]

    parts = split_text(prompt_question, ["<image>", "<speech>"])
    input_ids = []
    for part in parts:
        if part == "<image>":
            input_ids.append(IMAGE_TOKEN_INDEX)
        elif part == "<speech>":
            input_ids.append(SPEECH_TOKEN_INDEX)
        else:
            input_ids.extend(tokenizer(part).input_ids)

    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    image_tensor = [image[0][0].half()]
    image_sizes = [image[0][1]]
    generate_kwargs = {"eos_token_id": tokenizer.eos_token_id}

    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        speech=speech,
        speech_lengths=speech_lengths,
        do_sample=False,
        temperature=0.5,
        max_new_tokens=4096,
        modalities=["video"],
        **generate_kwargs,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    print(text_outputs)


if __name__ == "__main__":
    main()
