import json
from argparse import Namespace
from typing import List, Optional

import torch
from pydantic import BaseModel, Field

from slime.rollout.sglang_rollout import GenerateState, _load_and_encode_image
from slime.utils.http_utils import post
from slime.utils.types import Sample


# --- Pydantic Schemas from async_sglang_rollout.py ---
class TaskNoToolLV5(BaseModel):
    thought: str
    conclusion: str
    title: str

class TaskNoToolLV4(BaseModel):
    thought: str
    subtasks: Optional[List[TaskNoToolLV5]] = Field(default=None, max_items=5)
    conclusion: str
    title: str

class TaskNoToolLV3(BaseModel):
    thought: str
    subtasks: Optional[List[TaskNoToolLV4]] = Field(default=None, max_items=5)
    conclusion: str
    title: str

class TaskNoToolLV2(BaseModel):
    thought: str
    subtasks: Optional[List[TaskNoToolLV3]] = Field(default=None, max_items=5)
    conclusion: str
    title: str

class TaskNoTool(BaseModel):
    thought: str
    subtasks: Optional[List[TaskNoToolLV2]] = Field(default=None, max_items=5)
    conclusion: str
    title: str

TaskNoTool.model_rebuild()

class SolutionNoTool(BaseModel):
    reasoning: List[TaskNoTool]
    answer: str


async def generate(args: Namespace, sample: Sample, sampling_params: dict) -> Sample:
    """
    Custom generate function that enforces constrained decoding AND calculates entropy.
    This function replicates the logic from `sglang_rollout.generate` to access
    the raw server response.
    """
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert sample.status in {Sample.Status.PENDING, Sample.Status.ABORTED}, f"Sample status is {sample.status}"

    # --- Start of replicated logic from sglang_rollout.generate ---
    image_data = []
    if isinstance(sample.prompt, str):
        text_prompt = sample.prompt
    else:
        text_prompt = ""
        image_token = state.tokenizer.special_tokens_map.get("image_token", "<image>")
        for part in sample.prompt:
            if part["type"] == "text":
                text_prompt += part["text"]
            elif part["type"] == "image":
                text_prompt += image_token
                try:
                    img_b64 = await torch.hub.threading.run_in_executor(_load_and_encode_image, part["path"])
                    image_data.append(img_b64)
                except Exception as e:
                    print(f"Error processing image {part['path']}: {e}")
                    sample.status = Sample.Status.ABORTED
                    return sample

    if len(sample.response) > 0:
        prompt_len = len(state.tokenizer(text_prompt, add_special_tokens=False)["input_ids"])
        sampling_params["max_new_tokens"] -= len(sample.tokens) - prompt_len

    if sampling_params["max_new_tokens"] <= 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    # --- Payload Modification ---
    request_sampling_params = sampling_params.copy()
    #request_sampling_params["json_schema"] = json.dumps(SolutionNoTool.model_json_schema())
    payload = {
        "sampling_params": request_sampling_params, 
        "return_logprob": True,
        "top_logprobs_num": 20
    }
    if image_data:
        payload["image_data"] = image_data

    if len(sample.response) > 0:
        payload["input_ids"] = sample.tokens
    else:
        prompt_token_ids = state.tokenizer(text_prompt, add_special_tokens=False)["input_ids"]
        payload["input_ids"] = prompt_token_ids
        if not sample.tokens:
            sample.tokens = prompt_token_ids

    # --- Network Call ---
    output = await post(url, payload)

    # --- INJECTION POINT FOR ENTROPY CALCULATION ---
    if "output_token_logprobs" in output["meta_info"]:
        new_response_tokens, new_response_log_probs, new_entropies = [], [], []

        for item in output["meta_info"]["output_token_logprobs"]:
            new_response_log_probs.append(item[0])
            new_response_tokens.append(item[1])
            top_logprobs_dict = item[2]
            print(top_logprobs_dict)

            if top_logprobs_dict:
                log_probs = torch.tensor(list(top_logprobs_dict.values()), dtype=torch.float32)
                probs = torch.exp(log_probs)
                # Normalize probabilities to sum to 1 for the entropy calculation
                # This calculates entropy over the truncated top-k distribution
                probs /= probs.sum()
                entropy = -torch.sum(probs * torch.log(probs)).item()
                new_entropies.append(entropy)
            else:
                new_entropies.append(0.0)

        # Update sample with new data
        sample.tokens.extend(new_response_tokens)
        sample.response_length += len(new_response_tokens)
        sample.response += output["text"]

        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        sample.rollout_log_probs.extend(new_response_log_probs)

        if sample.entropy is None:
            sample.entropy = []
        sample.entropy.extend(new_entropies)

    if "weight_version" in output["meta_info"]:
        sample.weight_versions.append(output["meta_info"]["weight_version"])

    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    return sample
