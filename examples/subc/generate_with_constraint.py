import json
from argparse import Namespace
from typing import List, Optional

import math
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
    request_sampling_params["top_logprobs_num"] = 20
    payload = {
        "sampling_params": request_sampling_params,
        "return_logprob": True,
        "top_logprobs_num": 20,
        "return_text_in_logprobs": True,
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
    meta_info = output.get("meta_info", {})
    token_logprobs = meta_info.get("output_token_logprobs", [])
    topk_all_steps = meta_info.get("output_top_logprobs")

    def _entropy_from_topk(candidates: Optional[List[List]]) -> float:
        if not candidates:
            return 0.0
        logps = [cand[0] for cand in candidates if cand and cand[0] is not None and math.isfinite(cand[0])]
        if not logps:
            return 0.0
        m = max(logps)
        if not math.isfinite(m):
            return 0.0
        exps = [math.exp(lp - m) for lp in logps]
        z = sum(exps)
        if z <= 0:
            return 0.0
        probs = [val / z for val in exps]
        return -sum(p * math.log(p) for p in probs if p > 0)

    if token_logprobs:
        new_response_tokens, new_response_log_probs, new_entropies = [], [], []

        for step, item in enumerate(token_logprobs):
            lp, tok_id, tok_text = (item + [None, None, None])[:3]
            new_response_log_probs.append(lp)
            new_response_tokens.append(tok_id)
            candidates = topk_all_steps[step] if topk_all_steps and step < len(topk_all_steps) else None
            new_entropies.append(_entropy_from_topk(candidates))

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
