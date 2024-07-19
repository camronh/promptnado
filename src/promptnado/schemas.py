from pydantic.v1 import BaseModel, Field
from typing import List, Union, Optional
from uuid import UUID
from langchain.schema import BaseMessage
from langsmith import Client
from langsmith.schemas import Dataset


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


class Example(BaseModel):
    """An example of a prompt"""
    input: Union[str, List[BaseMessage]]  # A string or Langchain message list
    # The reference output for the example in the dataset
    reference_output: Optional[str] = None



class LangsmithDataset(BaseModel):
    """An already existing Langsmith Dataset"""
    dataset_name: Optional[str] = None
    dataset_id: Optional[UUID] = None
    input_messages_key: str = "inputs"
    reference_output_key: str = "output"
    args_key: str = "args"
    dataset: Optional[Dataset] = None

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_dataset()

    def _initialize_dataset(self):
        client = Client()
        try:
            if self.dataset_name:
                self.dataset = client.read_dataset(dataset_name=self.dataset_name)
            elif self.dataset_id:
                self.dataset = client.read_dataset(dataset_id=self.dataset_id)
            elif not self.dataset:
                raise ValueError("Must provide either dataset_name or dataset_id")
        except Exception as e:
            print(f"Error initializing dataset: {e}")
            raise e