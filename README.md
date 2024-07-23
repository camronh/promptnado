# 🌪️ Promptnado: Your AI-Powered Prompt Engineer

Meet Promptnado – your personal AI prompt engineer! 🚀 By harnessing the power of AI, Promptnado acts as your personal AI prompt engineer, automatically generating, testing, and refining prompts to meet your specific criteria. Say goodbye to manual prompt tweaking and hello to AI-driven prompting!

## 🌟 Features

- **AI-powered iteration** automatically generates and refines prompts.
- Use **natural language instructions** to define your prompt criteria.
- **Automated evaluation** tests each generated prompt against your specifications.
- Generate **synthetic examples** to boost your testing dataset.
- Choose your **preferred models** for prompt generation, evaluation, and testing.
- Evaluate not just text outputs, but also the model's ability to make correct **function calls**.
- Support for **diverse input formats** including strings, input-output pairs, Langchain messages, and interpolated dictionaries.
- Create new datasets on the fly or use your own for testing.

## How Promptnado Works

1. You provide a system prompt with the <HERE> token to signify where our prompt changes will go.
2. You provide example requests to test these changes, or promptnado will generate them for you.
3. Promptnado's AI generates multiple prompt variations based on your instruction.
4. Each prompt is tested against your examples and evaluated using AI.
5. The process repeats, refining prompts until the evaluation results all pass.
6. You get a finely-tuned prompt that meets your specific needs!

## Installation

```bash
pip install promptnado
```

## Quick Start

```python
from promptnado import Promptnado

# The system prompt we want to optimize
# We set the <HERE> token to signify where our prompt changes will go
system_prompt = """You are a helpful assistant. 

Rules:
- You are only allowed to talk about coding
- <HERE>
- Try to be concise"""

# The goal of the prompt changes
instruction = "The agent should only respond in English."

# Let's set 2 examples to start
examples = ["¿Cómo estás?", "How do typescript generics work?"]

pn = Promptnado(system_prompt, instruction, examples, max_attempts=5)

# Generate 2 more examples
pn.generate_examples(count=2)

# Run the optimization
pn.run()
```


## Why Promptnado?

Prompt engineering is an art and a science. Promptnado brings the power of AI to this process, allowing you to **save time** on manual prompt tweaking, **discover optimal prompts** you might never have thought of, ensure **consistency and quality** in your AI interactions, and **adapt quickly** to new tasks and requirements.

Whether you're a seasoned AI engineer or just getting started with language models, Promptnado empowers you to create more effective, targeted prompts with ease. It's like having an AI prompt engineer right at your fingertips
