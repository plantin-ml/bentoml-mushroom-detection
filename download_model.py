import click
import shutil
from clearml import Task, Model


@click.command()
@click.option(
    "--task_id",
    help="ID of task",
)
def export(task_id):
    prev_task = Task.get_task(task_id=task_id)
    last_snapshot: Model = prev_task.models['output'][-1]
    shutil.copy(last_snapshot.get_local_copy(), './weights')

    click.secho(f"\n\nSuccess download model", fg="green")

if __name__=="__main__":
    export()
