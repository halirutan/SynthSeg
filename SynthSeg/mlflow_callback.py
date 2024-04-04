import mlflow
from mlflow.tensorflow.callback import MLflowCallback 
import configparser
from .training_options import TrainingOptions

from typing import Union, Optional

import os

class MLflowCustomCallback(MLflowCallback):
    def __init__(self, log_freq: Union[None, str, int] = "epoch", metric_type: Optional[str] = None, opts:Optional[TrainingOptions] = None):
        if log_freq == "epoch":
            kwargs = dict(log_every_epoch = True,log_every_n_steps=None)
        elif log_freq == "batch":
            kwargs = dict(log_every_epoch = False,log_every_n_steps=1)
        elif isinstance(log_freq, int):
            kwargs = dict(log_every_epoch = False,log_every_n_steps=log_freq)
        else: 
            kwargs = {}
        
        super().__init__(**kwargs)
        self._chief_worker_only = True
        self.opts = opts
        self.metric_type = metric_type

            
    def on_train_begin(self, logs=None):
        super().on_train_begin(logs= logs)
        mlflow.log_params({key: item for key, item in os.environ.items() if "SLURM" in key or "SBATCH" in key})
        mlflow.log_params(self.opts.__dict__)
        
        
    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at the end of each epoch."""
        if not self.log_every_epoch or logs is None:
            return
        
        
        metrics = {f"train_{k}" if not any(k.startswith(prefix) for prefix in ["val_", "test_", "train_"]) else k: v for k, v in self.transform_logs(logs).items()}
        mlflow.log_metrics(metrics, step=epoch, synchronous=False)


    def on_train_batch_end(self, batch, logs=None):
        """Log metrics at the end of each batch with user specified frequency."""
        if self.log_every_n_steps is None or logs is None:
            return
        current_iteration = int(self.model.optimizer.iterations.numpy())

        if current_iteration % self.log_every_n_steps == 0:
            
            metrics = {"train_" + k: v for k, v in self.transform_logs(logs).items()}
            mlflow.log_metrics(metrics, step=current_iteration, synchronous=False)


    def on_test_end(self, logs=None):
        
        # """Log validation metrics at validation end."""
        if self.log_every_epoch or logs is None:
            return
        current_iteration = int(self.model.optimizer.iterations.numpy())
        
        
        metrics = {"val_" + k: v for k, v in self.transform_logs(logs).items()}
        mlflow.log_metrics(metrics,  step=current_iteration, synchronous=False)
    
        
    def transform_logs(self, logs):
        return {f"{k}-{self.metric_type}" if ("loss" in k) else k: v for k, v in logs.items()} if self.metric_type is not None else logs
       