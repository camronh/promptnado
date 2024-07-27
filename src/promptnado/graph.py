from langgraph.graph import StateGraph, START, END
from typing import List, TypedDict
from .schemas import Rule


class State(TypedDict):
    current_rule: Rule                 # The rule currently being tested
    current_prompt: str               # The prompt associated with the current rule
    instructions: str               # Instructions for the promptnado
    # List of rules that have not been tested yet
    untested_rules: List[Rule]
    solved: bool                   # Indicates whether the current rule has solved the problem
    attempt: int               # The number of attempts made with the current rule
    successful_prompt: str  # The prompt that was successful, if any
    tested_rules: List[Rule]           # List of rules that have been tested
    # test_results: List[str]           # Results of each test performed


class Nodes:
    def __init__(self, pn):
        self.pn = pn

    # Nodes
    def generate_rules(self, state: State):
        self.pn._generate_rules()
        return {"untested_rules": self.pn.rules}

    def next_rule(self, state: State):
        """Set the state to be ready to test the next rule"""
        self.pn._next_rule()
        return {"current_rule": self.pn.current_rule, "current_prompt": self.pn.current_prompt, "untested_rules": self.pn.rules}

    def test_rule(self, state: State):
        # Pop rule from state["rules"]
        current_rule = state["current_rule"]
        results = self.pn._test_rule(current_rule)
        return {"solved": self.pn.solved, "current_prompt": self.pn.current_prompt, "attempt": self.pn.attempts}

    def success(self, state: State):
        print("\n\nSolved!! Current prompt can be found at `self.successful_prompt`\n\n")
        # ANSI escape codes for green text and reset
        GREEN = "\033[92m"
        RESET = "\033[0m"

        green_rule = f"{GREEN}{self.pn.current_rule.prompt}{RESET}"
        highlighted_prompt = self.pn.system_prompt.replace(
            self.pn.rule_token, green_rule)

        print(
            f"Successful prompt:\n====================\n{highlighted_prompt}\n=================")
        print(self.pn.dataset.dataset.url)
        return {"current_rule": self.pn.current_rule, "successful_prompt": self.pn.successful_prompt, "solved": True, "attempts": self.pn.attempts}

    def failure(self, state: State):
        return {"tested_rules": self.pn.tested_rules, "solved": False, "attempts": self.pn.attempts}

    # edges
    def route(self, state: State):
        if self.pn.solved:
            return "success"

        # If weve reached the max attempts route to failure
        if self.pn.attempts >= self.pn.max_attempts:
            return "failure"

        # If we have no more rules left, generate more
        if not self.pn.rules:
            return "generate_rules"

        return "next"


def create_graph(pn):
    graph = StateGraph(State)
    nodes = Nodes(pn)

    # Add nodes
    graph.add_node("GenerateRules", nodes.generate_rules)
    graph.add_node("NextRule", nodes.next_rule)
    graph.add_node("TestRule", nodes.test_rule)
    graph.add_node("Success", nodes.success)
    graph.add_node("Failure", nodes.failure)

    # Add edges
    graph.add_edge("GenerateRules", "NextRule")
    graph.add_edge("NextRule", "TestRule")

    # Conditional edges from test_rule
    graph.add_conditional_edges(
        "TestRule",
        nodes.route,
        {
            "success": "Success",
            "generate_rules": "GenerateRules",
            "next": "NextRule",
            "failure": "Failure"
        }
    )

    # Set the entry point
    graph.set_entry_point("GenerateRules")
    app = graph.compile()
    return app
