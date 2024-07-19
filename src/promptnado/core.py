from typing import List, Union
import random
from langsmith.schemas import Run, Example as DatasetExample
from langsmith import evaluate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

from langsmith import Client
client = Client()
from langchain.schema import SystemMessage, HumanMessage, BaseMessage, AIMessage
from .schemas import Rule, Rules, CorrectnessEvaluationResult, Example


class Promptnado:
    def __init__(self, system_prompt: str, instruction: str, examples: List[Union[str, Example, List[BaseMessage]]],
                 rule_token="<HERE>", max_attempts=10,
                 rule_gen_model=init_chat_model(
                     "gpt-4o-mini", temperature=0.7),
                 eval_model=init_chat_model("gpt-4o-mini", temperature=0.7),
                 prediction_model=init_chat_model("gpt-4o-mini", temperature=0.7)):

        # rule_token is not in the prompt throw
        if rule_token not in system_prompt:
            raise ValueError(
                f"Rule token {rule_token} not found in system prompt")

        self.system_prompt = system_prompt
        self.instruction = instruction
        self.examples = examples
        self.rule_token = rule_token

        # Create random dataset name
        self.dataset_name = f"Promptnado_{random.randint(0, 1000000)}"

        self.attempts = 1
        self.solved = False
        self.current_rule = None
        self.current_prompt = None
        self.successful_prompt = None
        self.rule_gen_model = rule_gen_model
        self.eval_model = eval_model
        self.prediction_model = prediction_model
        self.max_attempts = max_attempts

    def _create_dataset(self):
        """Create a dataset with a unique name"""
        dataset = client.create_dataset(
            self.dataset_name, description=self.instruction)
        for example in self.examples:
            if isinstance(example, str):
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
            f"Created dataset: {self.dataset_name} with {len(self.examples)} examples")
        print(dataset.url)

    def _generate_rules(self):
        """Use an LLM to generate a list of rules"""

        system_prompt = f"""You are an expert LLM Prompt Engineer. Your job is to try to solve for the provided <Instructions> \
by making adjustments to the <Original Prompt>. You should attempt to make 5 suggestions for prompts that might work. Each suggestion you \
make will be interpolated into the prompt where {self.rule_token} is, and then evaluated for correctness against a dataset of \
examples.

<Instructions>
{self.instruction}
</Instructions>

<Original Prompt>
{self.system_prompt}
</Original Prompt>
"""

        structured_llm = self.rule_gen_model.with_structured_output(Rules)

        rules: Rules = structured_llm.invoke(system_prompt)

        self.rules = rules.rules
        print(f"Generated {len(self.rules)} rules\n")
        print(self.rules)
        return self.rules

    def _build_prompt(self, rule: Rule):
        """Interpolate the rules into the system prompt"""
        interpolated_prompt = self.system_prompt.replace(
            self.rule_token, rule.prompt)

        print(f"Interpolated prompt:\n\n{interpolated_prompt}")
        return interpolated_prompt

    def _evaluate_correctness(self, run: Run, example: DatasetExample):
        """Eval function to use an LLM to validate that the instruction was followed"""

        # Add reference output if it exists
        if example.outputs and example.outputs.get("output"):
            reference_example = f"""\n<Example Output>
{example.outputs["output"]}
</Example Output>\n"""
        else:
            reference_example = ""

        system_prompt = f"""Your job is to validate whether the <Result> meets the criteria for <Instruction>. Try to be a harsh judge.

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
            if isinstance(inputs["inputs"], str):
                messages = [
                    SystemMessage(content=self.current_prompt),
                    HumanMessage(content=inputs["inputs"]),
                ]

            elif isinstance(inputs["inputs"], list):
                messages = [SystemMessage(content=self.current_prompt)]
                for msg_dict in inputs["inputs"]:
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
            print(f"Error predicting: {e}")
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

        self.current_prompt = self._build_prompt(self.current_rule)

        results = evaluate(
            self._predict,
            data=self.dataset_name,
            evaluators=[self._evaluate_correctness],
            experiment_prefix=f"Attempt-{self.attempts}",
        )

        self.attempts += 1

        return results

    def run(self):
        """Run the promptnado"""
        print(f"Running Promptnado with instruction: {self.instruction}")
        self._create_dataset()

        while not self.solved and self.attempts < self.max_attempts:
            try:
                self._generate_rules()
                for rule in self.rules:
                    if self.attempts > self.max_attempts:
                        print("Max attempts reached")
                        return
                    results = self._test_rule(rule)
                    if self._is_solved(results):
                        self.results = results
                        self.solved = True
                        self.successful_prompt = self.current_prompt
                        break
            except Exception as e:
                print(f"Fatal error encountered: {e}")
                return  # Exit the while loop on error

        print("\n\nSolved!! Current prompt can be found at `self.successful_prompt`\n\n")

        print(
            f"Successful prompt:\n====================\n{self.current_prompt}\n=================")
        print(self.results._manager._experiment.url)