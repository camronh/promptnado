
import typer
from promptnado.core import Promptnado
from typing import Optional

app = typer.Typer()

@app.command()
def run(
    system_prompt: str = typer.Option(..., help="The system prompt to use."),
    instruction: str = typer.Option(..., help="Instruction for the prompt."),
    rule_token: str = typer.Option("<HERE>", help="Token to indicate rule insertion point."),
    max_attempts: int = typer.Option(10, help="Maximum number of attempts."),
    experiment_name: Optional[str] = typer.Option(None, help="Name of the experiment."),
    max_concurrency: Optional[int] = typer.Option(None, help="Maximum number of concurrent operations.")
):

    # Create an instance of the Promptnado class with the provided parameters
    promptnado_instance = Promptnado(
        system_prompt=system_prompt,
        instruction=instruction,
        rule_token=rule_token,
        max_attempts=max_attempts,
        experiment_name=experiment_name,
        max_concurrency=max_concurrency
    )

    # Run the main functionality
    promptnado_instance.run()