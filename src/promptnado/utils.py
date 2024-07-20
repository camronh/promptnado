from typing import List, Optional, Type
from pydantic.v1 import BaseModel, Field, create_model
from langchain.chat_models import init_chat_model
from .schemas import Example


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
    input: str = Field(
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
                      count: int = 3):
    """Generate Synthetic Examples Schema"""
    Schema = generate_examples_schema(arg_schema)

    system_prompt = f"""You are a synthetic data generator. We will be testing an LLM's ability to follow the instructions in \
<System Prompt>. Your job is to generate a list of EXACTLY {count} examples that would challenge the LLM's ability to follow the \
instructions in <Instructions> to the tee.

GUIDELINES:
- Generate EXACTLY {count} examples.
- Try to make the examples as diverse as possible
- Your goal should be to generate examples that will challenge the LLM's ability to follow the instructions in <Instructions>. Not testing the LLM's\
general ability, or ability to follow the <System Prompt>. We are specifically looking to see if it is able to follow the <Instructions>. The \
<System Prompt> is just for added context to generate better examples.
- The LLM's outputs will be evaluated ONLY based on the <Instructions> so make sure the examples are directly relevant to the instructions.
- Include at least one "Happy path" example that is less challenging than the others.

<Instructions>
{instructions}
</Instructions>

<System Prompt>
{prompt}
</System Prompt>
"""

    llm_with_tools = init_chat_model(
        "gpt-4o-mini", temperature=0.7).with_structured_output(Schema)

    results: Schema = llm_with_tools.invoke(system_prompt)
    print(results)
    if not arg_schema:
        return [Example(input=result.input, reference_output=result.reference_output) for result in results.examples]
    else:
        return [result.dict() for result in results.examples]
