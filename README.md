# Promptnado

Promptnado is a framework for iterating on system prompts using evaluations. It automates the process of generating, testing, and refining prompts to meet specific criteria.

## Features

- Automated rule generation for system prompts
- Evaluation of generated prompts against user-defined criteria
- Iteration until desired performance is achieved
- Integration with LangChain and LangSmith for robust language model interactions

## Installation

You can install Promptnado using pip:

```bash
pip install promptnado
```

## Quick Start

Here's a basic example of how to use Promptnado:

```python
from promptnado import Promptnado

# Define your system prompt, instruction, and examples
system_prompt = """You are a helpful assistant. 

Rules:
- You are only allowed to talk about coding
- <HERE>
- Try to be concise"""

instruction = "The agent should only respond in English."

examples = ["Hi there!", "Como estas?", "What's your favorite programming language?"]

# Create a Promptnado instance
pn = Promptnado(system_prompt, instruction, examples, max_attempts=5)

# Run the prompt optimization
pn.run()

# Get the optimized prompt
optimized_prompt = pn.successful_prompt
print(f"Optimized prompt:\n{optimized_prompt}")
```
