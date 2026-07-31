import mlflow

class MLflowLogger():
  def __init__(self, experiment_name: str):
    print("Tracking URI:", mlflow.get_tracking_uri())
    mlflow.set_experiment(experiment_name=experiment_name)

  def start_run(self, run_name=None):
    mlflow.start_run(run_name=run_name)

  def log_params(self, params: dict):
    mlflow.log_params(params=params)

  def log_metrics(self, metrics: dict, step=None):
    mlflow.log_metrics(metrics=metrics, step=step)

  def log_artifacts(self, path):
    mlflow.log_artifacts(path)

  def end_run(self):
    mlflow.end_run()