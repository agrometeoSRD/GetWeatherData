"""Console script for getweatherdata."""
# not actually used, but will be useful for scripts that have CLI

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for getweatherdata."""
    console.print("Replace this message by putting your code into "
               "getweatherdata.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")



if __name__ == "__main__":
    app()
