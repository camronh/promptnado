from .schemas import Rule, Rules, CorrectnessEvaluationResult, Example, LangsmithDataset
from langchain.schema import SystemMessage, HumanMessage, BaseMessage, AIMessage
from langsmith import Client
from typing import List, Union
import random
from langsmith.schemas import Run, Example as DatasetExample
from langsmith import evaluate
from langchain.chat_models import init_chat_model
from .utils import format_rules, generate_examples
from dotenv import load_dotenv
from .graph import create_graph
load_dotenv()

client = Client()


class Promptnado:
    def __init__(self, system_prompt: str, instruction: str,
                 examples: List[Union[str, dict,
                                      Example, List[BaseMessage]]] = [],
                 rule_token="<HERE>", max_attempts=10,
                 rule_gen_model=init_chat_model(
                     "gpt-4o-mini", temperature=0.7),
                 eval_model=init_chat_model("gpt-4o-mini", temperature=0.7),
                 prediction_model=init_chat_model(
                     "gpt-4o-mini", temperature=0.7),
                 dataset: LangsmithDataset = None,
                 experiment_name: str = None,
                 max_concurrency: int = None):

        # rule_token is not in the prompt throw
        if rule_token not in system_prompt:
            raise ValueError(
                f"Rule token {rule_token} not found in system prompt")

        self.system_prompt = system_prompt
        self.instruction = instruction
        self.examples = examples
        self.rule_token = rule_token
        self.tested_rules = []
        self.max_concurrency = max_concurrency

        if dataset:
            self.dataset = dataset
            self.dataset_name = dataset.dataset.name
        else:
            # Create random dataset name
            self.dataset_name = f"{experiment_name or 'Promptnado'}_{random.randint(0, 1000000)}"
            self.dataset = None

        self.attempts = 0
        self.solved = False
        self.current_rule = None
        self.current_prompt = None
        self.successful_prompt = None
        self.rule_gen_model = rule_gen_model
        self.eval_model = eval_model
        self.prediction_model = prediction_model
        self.max_attempts = max_attempts
        self.app = create_graph(self)

    def generate_examples(self, count=3, arg_schema=None):
        """Generate Synthetic examples and add them to the `examples` list"""
        examples = generate_examples(
            self.system_prompt, self.instruction, count=count, arg_schema=arg_schema)
        self.examples.extend(examples)
        return examples

    def _create_dataset(self):
        """Create a dataset with a unique name"""
        dataset = client.create_dataset(
            self.dataset_name, description=self.instruction)

        examples = self.examples if self.examples else [""]

        for example in examples:
            if isinstance(example, dict):
                client.create_example(
                    inputs={"inputs": {"args": example}}, dataset_id=dataset.id)
            elif isinstance(example, str):
                client.create_example(
                    inputs={"inputs": example}, dataset_id=dataset.id)
            elif isinstance(example, Example):
                inputs = example.input
                if isinstance(inputs, str):
                    example_inputs = {"inputs": inputs}
                elif isinstance(inputs, list) and all(isinstance(msg, BaseMessage) for msg in inputs):
                    example_inputs = {"inputs": inputs}
                else:
                    raise ValueError(
                        "Invalid input format in Example. Must be a string or a list of BaseMessages.")

                example_data = {"inputs": example_inputs}
                if example.reference_output:
                    example_data["outputs"] = {
                        "output": example.reference_output}

                client.create_example(**example_data, dataset_id=dataset.id)
            elif isinstance(example, list) and all(isinstance(msg, BaseMessage) for msg in example):
                client.create_example(
                    inputs={"inputs": example}, dataset_id=dataset.id)
            else:
                raise ValueError(
                    f"Invalid example format. Must be a string, Example, or a list of BaseMessages.\nActual Type: {type(example)}")

        print(
            f"Created dataset: {self.dataset_name} with {len(examples)} examples")
        print(dataset.url)
        self.dataset = LangsmithDataset(
            dataset_name=dataset.name, dataset_id=dataset.id, dataset=dataset)
        return self.dataset

    def _generate_rules(self):
        """Use an LLM to generate a list of rules"""

        system_prompt = f"""You are an expert LLM Prompt Engineer. Your job is to try to solve for the provided <Instructions> \
by making adjustments to the <Original Prompt>. You should attempt to make up to 5 suggestions for prompts that might work. Each suggestion you \
make will be interpolated into the prompt where {self.rule_token} is, and then evaluated for correctness against a dataset of \
examples.

GUIDELINES:
- You must make up to 5 suggestions for prompts that might work.
- Gradually increase the complexity of your suggestions. Your first suggestion should be the simplest one that you think might work, and your last suggestion should be the most complex one that you think might work.
- Your reasoning should explain why you think this prompt would work and what makes it different from your other attempts.
- You must provide a prompt for each suggestion.
- Try to generate a diverse set of rules. Each one should be different from the others.
- Only make adjustments to where you see {self.rule_token} in the <Original Prompt>. No other part of the prompt will be changed.
- Your `prompt` will be interpolated to the {self.rule_token} verbatim, so if there is formatting to be accounted for, make sure to include it.
- If some other prompt <ATTEMPTS> are included, try to go a different direction than the previous attempts unless you can't think of a new one. In that case, try to improve on the previous attempts.
- Try to keep your reasoning pretty short. No need to be too detailed.
- MAKE SURE your prompts are specific to the <Instructions>. Be careful to avoid adding any instructions that may effect behavior not mentioned in the <Instructions>. For example, don't mention conciseness if its not explicitly mentioned in the <Instructions>.



PROMPTING TIPS:
- Simple is almost always better. Only if a simple option doesn't work should you consider more complex options.
- Capitalization can be used to put emphasis on certain words. For example "NEVER" or "ALWAYS".
- Providing an example can help with some subtleties. If you ever need to show what you mean instead of just saying it, a very brief inline example can help.
- Consider the entire <Original Prompt> when making your suggestions, there might be some other parts of the prompt that may play a part in the behavior we are expecting. 

<Instructions>
{self.instruction}
</Instructions>

{format_rules(self.tested_rules) if self.tested_rules else ""}

<Original Prompt>
{self.system_prompt}
</Original Prompt>
"""

        structured_llm = self.rule_gen_model.with_structured_output(Rules)

        rules: Rules = structured_llm.invoke(system_prompt)

        self.rules = rules.rules
        # self.tested_rules.extend(rules.rules)
        print(f"Generated {len(self.rules)} rules\n")
        print(self.rules)
        return self.rules

    def _next_rule(self):
        """Get the next rule to test"""
        self.current_rule = self.rules.pop(0)
        self.current_prompt = self._build_prompt(self.current_rule)

    def _build_prompt(self, rule: Rule):
        """Interpolate the rules into the system prompt"""
        interpolated_prompt = self.system_prompt.replace(
            self.rule_token, rule.prompt)

        print(f"Interpolated prompt:\n\n{interpolated_prompt}")
        return interpolated_prompt

    def _evaluate_correctness(self, run: Run, example: DatasetExample):
        """Eval function to use an LLM to validate that the instruction was followed"""

        # Add reference output if it exists
        if example.outputs and example.outputs.get(self.dataset.reference_output_key):
            reference_example = f"""\n<Example Output>
{example.outputs[self.dataset.reference_output_key]}
</Example Output>\n"""
        else:
            reference_example = ""

        system_prompt = f"""Your job is to validate whether the <Result> meets the criteria for <Instruction>. Try to be a harsh judge. \
If you are not sure, try to be conservative and say that the result does not meet the criteria, and explain why.

<Instruction>
{self.instruction}
</Instruction>
{reference_example}
<Result>
{run.outputs["output"]}
</Result>
"""

        structured_llm = self.eval_model.with_structured_output(
            CorrectnessEvaluationResult)

        result: CorrectnessEvaluationResult = structured_llm.invoke(
            system_prompt)

        return {"score": 1 if result.correct else 0, "key": "correctness", "comment": result.reasoning}

    def _predict(self, inputs: dict):
        """Run current prompt against example in the dataset"""
        try:
            input_key = self.dataset.input_messages_key
            if isinstance(inputs[input_key], dict):
                messages = [SystemMessage(
                    content=self.current_prompt.format(**inputs[input_key][self.dataset.args_key]))]

            elif isinstance(inputs[input_key], str):
                messages = [SystemMessage(content=self.current_prompt)]
                if inputs[input_key] != "":  # If empty input, dont add it as a message
                    messages.append(HumanMessage(content=inputs[input_key]))

            elif isinstance(inputs[input_key], list):
                messages = [SystemMessage(content=self.current_prompt)]
                for msg_dict in inputs[input_key]:
                    if isinstance(msg_dict, dict):
                        if msg_dict["type"] == "human":
                            messages.append(HumanMessage(**msg_dict))
                        elif msg_dict["type"] == "ai":
                            messages.append(AIMessage(**msg_dict))
                        else:
                            messages.append(BaseMessage(**msg_dict))
                    elif isinstance(msg_dict, BaseMessage):
                        messages.append(msg_dict)
                    else:
                        raise ValueError(f"Invalid message format: {msg_dict}")
            else:
                raise ValueError("Invalid input format")

            # Invoke the model
            response = self.prediction_model.invoke(messages)
            if response.tool_calls:
                return {"output": f"Tool Calls:\n{response.tool_calls}"}

            return {"output": response.content}

        except Exception as e:
            raise e

    def _is_solved(self, eval_results):
        """Validate the results"""

        results = eval_results._results

        # If any of the result scores are not a 1, return false
        if len(results) == 0:
            raise Exception("No results found")

        for result in results:
            score = result['evaluation_results']["results"][0].score
            if score != 1:
                return False

        return True

    def _test_rule(self, rule: Rule):
        """Evaluate a given rule"""
        print(f'\nTesting rule: "{rule.prompt}"')
        self.current_rule = rule

        self.current_prompt = self.current_prompt

        results = evaluate(
            self._predict,
            data=self.dataset.dataset.name,
            evaluators=[self._evaluate_correctness],
            experiment_prefix=f"Attempt-{self.attempts}",
            max_concurrency=self.max_concurrency
        )

        self.attempts += 1

        if self._is_solved(results):
            self.solved = True
            self.results = results
            self.successful_prompt = self.current_prompt

        return results

    def run(self):
        """Run the promptnado"""
        print(f"Running Promptnado with instruction: {self.instruction}")
        if not self.dataset:
            self._create_dataset()
        else:
            print(f"Using existing dataset: {self.dataset.dataset.name}")
            print(self.dataset.dataset.url)

        output = self.app.invoke({"instructions": self.instruction})

        return output
