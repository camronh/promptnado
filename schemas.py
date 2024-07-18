from pydantic.v1 import BaseModel, Field
from typing import List


class Rule(BaseModel):
    """A single rule for the prompt"""
    reasoning: str = Field(
        ..., description="The thought process and direction for why we think this is a good solution to the instruction.")
    prompt: str = Field(..., description="A single prompt rule that we can try to solve for the instruction. \
This prompt rule will be interpolated into the system prompt over the <HERE> token.")


class Rules(BaseModel):
    """Set of prompt rules that should be tried"""
    rules: List[Rule] = Field(
        ..., description="A list of prompt rules that we can try to solve for the instruction.")


class CorrectnessEvaluationResult(BaseModel):
    """Result of an evaluation of correctness"""
    reasoning: str = Field(
        ..., description="The thought process behind why you think the answer is correct or incorrect.")
    correct: bool = Field(..., description="Correctness score")
