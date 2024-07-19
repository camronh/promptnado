from typing import List, Optional, Type
from pydantic.v1 import BaseModel, Field, create_model


def format_rules(rules):
    rule_strs = [
        f"<ATTEMPT>\nReasoning: {rule['reasoning']}\nPrompt: '{rule['prompt']}'\n</ATTEMPT>" for rule in rules]

    attempts_str = '\n'.join(rule_strs)

    return f"""Here are some prompts we have already tried:
    
<ATTEMPTS>
{attempts_str}
</ATTEMPTS>
"""


class MessageExample(BaseModel):
    input_message: str = Field(
        description="The human message input to be used in the example in the dataset. We will run this message against the System prompt message.")
    reference_output: str = Field(
        description="An output that would be expected. It needs to satisfy the <Instructions>.")


def generate_examples_schema(arg_schema: Optional[Type[BaseModel]] = None) -> Type[BaseModel]:
    """Generate a schema for Synthetic Examples"""

    # Determine which schema to use for the examples
    ExampleSchema = arg_schema if arg_schema else MessageExample

    # Create the SyntheticExamples schema dynamically
    SyntheticExamples = create_model(
        "SyntheticExamples",
        examples=(List[ExampleSchema], Field(
            description="A list of examples to be added to a dataset, which will eventually be used for evaluating the prompt for adherence to the <Instructions>."
        )),
    )

    return SyntheticExamples


def generate_examples(prompt: str, instructions: str,
                      arg_schema: Optional[Type[BaseModel]] = None,
                      count: int = 3) -> Type[BaseModel]:
    """Generate Synthetic Examples Schema"""
    Schema = generate_examples_schema(arg_schema)

    

