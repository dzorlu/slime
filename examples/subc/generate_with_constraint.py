import json
from argparse import Namespace
from typing import List, Optional

from pydantic import BaseModel, Field

# Import the original `generate` function to reuse its HTTP client logic
from slime.rollout.sglang_rollout import generate as default_sglang_generate
from slime.utils.types import Sample


# --- Pydantic Schemas from async_sglang_rollout.py ---

class TaskNoToolLV5(BaseModel):
    thought: str
    conclusion: str
    title: str

class TaskNoToolLV4(BaseModel):
    thought: str
    subtasks: Optional[List[TaskNoToolLV5]] = Field(
        default=None,
        max_items=5
    )
    conclusion: str
    title: str

class TaskNoToolLV3(BaseModel):
    thought: str
    subtasks: Optional[List[TaskNoToolLV4]] = Field(
        default=None,
        max_items=5
    )
    conclusion: str
    title: str

class TaskNoToolLV2(BaseModel):
    thought: str
    subtasks: Optional[List[TaskNoToolLV3]] = Field(
        default=None,
        max_items=5
    )
    conclusion: str
    title: str

class TaskNoTool(BaseModel):
    thought: str
    subtasks: Optional[List[TaskNoToolLV2]] = Field(
        default=None,
        max_items=5
    )
    conclusion: str
    title: str

TaskNoTool.model_rebuild()

class SolutionNoTool(BaseModel):
    reasoning: List[TaskNoTool]
    answer: str


async def generate(
    args: Namespace,
    sample: Sample,
    sampling_params: dict
) -> Sample:
    """
    A custom generate function that wraps the default SGLang generate function
    to enforce constrained decoding with a JSON schema.
    """
    # Create a copy of the sampling params to avoid modifying the original.
    request_sampling_params = sampling_params.copy()

    # Inject the `json_schema` from the complex SolutionNoTool Pydantic model.
    request_sampling_params["json_schema"] = json.dumps(SolutionNoTool.model_json_schema())

    # Call the original `generate` function with the modified parameters.
    updated_sample = await default_sglang_generate(
        args,
        sample,
        request_sampling_params
    )

    return updated_sample
