import typer
from rich.console import Console
from rich.table import Table

from ..kernels import REGISTRY

app = typer.Typer(
    name="kompass",
    add_completion=True,
)
console = Console()


@app.command(name="list")
def list_ops():
    table = Table(title="Kompass Registry")
    table.add_column("Kernels", style="magenta")
    for op_name, impls in REGISTRY.items():
        table.add_row(op_name, str(impls))

    console.print(table)


if __name__ == "__main__":
    app()
