"""
Created on 29.08.21
@author :ali
"""

import torch
from ignite.engine import Engine
from ignite.engine import Events
from ignite.handlers import EarlyStopping
from ignite.utils import setup_logger
from torch.optim.lr_scheduler import ExponentialLR


class EEGBasedTrainer():
    def __init__(self, model, optimizer, loss, metrics, args, log_dir, device=torch.device("cuda"),
                 batch_transform=None):

        self.optimizer = optimizer
        self.loss_fn = loss
        self.loss = loss
        self.metrics = metrics
        self.args = args
        self.log_dir = log_dir
        self.device = device
        if args.tunning.enable_optuna_tuner:
            self.trial = None

        self.train_engine = None
        self.eval_engine = None
        self.batch_transform = batch_transform
        self.model = model

        if args.network.model_path is not None:
            try:
                self.model.load_state_dict(torch.load(args.network.model_path))
            except FileNotFoundError:
                print("checkpoint model not found, please check the model path")

    @staticmethod
    def attach_metrics(engine, metrics):
        if not isinstance(metrics, dict):
            raise RuntimeError("Metrics must be a dictionary")
        for name, metric in metrics.items():
            metric.attach(engine, name)

    def update_trial(self, trial):
        self.trial = trial

    def _compute_batch(self, batch, eval_mode=False, non_blocking=False):

        if self.batch_transform is not None:
            batch = self.batch_transform(batch)
        if eval_mode:
            self.model.eval()
            torch.set_grad_enabled(False)
        else:
            self.model.train()
        if eval_mode:
            pass

        x, y = batch
        x = x.to(self.device, non_blocking=non_blocking)
        y = y.to(self.device, non_blocking=non_blocking)

        if not eval_mode:
            self.optimizer.zero_grad()
        y_pred = self.model(x)
        # y_pred = torch.squeeze(y_pred, 1)
        # y = torch.unsqueeze(y, dim=1)
        loss = self.loss_fn(y_pred, y)

        # if len(out) == 2:
        #     loss, y_pred = out
        #     kwargs = {}
        # else:
        #     loss, y_pred, kwargs = out

        if not eval_mode:
            loss.backward()  # loss is first item in output
            self.optimizer.step()
        else:
            torch.set_grad_enabled(True)

        d = {"loss": loss.item()}

        return y_pred, y, d

    def _create_engine(self, eval_mode=False, non_blocking=False):
        # wrapper to pass kwargs
        def _inference_wrapper(engine, batch):
            return self._compute_batch(engine, batch, eval_mode=eval_mode, non_blocking=non_blocking)

        engine = Engine(_inference_wrapper)
        return engine

    def create_train_engine(self, name="train_engine", metric_names=None, non_blocking=None):
        self.train_engine = self._create_engine(eval_mode=False, non_blocking=non_blocking)

        if metric_names is not None:
            metrics = {kk: self.metrics[kk] for kk in metric_names}
            self.attach_metrics(self.train_engine, metrics)
        self.train_engine.name = name
        return self.train_engine

    def create_eval_engine(self, name="eval_engine", metric_names=None, non_blocking=False):
        self.eval_engine = self._create_engine(eval_mode=True, non_blocking=non_blocking)

        if metric_names is not None:
            metrics = {kk: self.metrics[kk] for kk in metric_names}
            self.attach_metrics(self.eval_engine, metrics)
        self.eval_engine.name = name
        return self.eval_engine

    def run_evaluator(self, data_loader):
        if self.eval_engine is None:
            raise RuntimeError("Evaluator engine does not exist. You have to create an evaluator with "
                               "obj.create_evaluator at first")
        self.eval_engine.run(data_loader, max_epochs=1)

    def run_trainer(self, train_data_loader, num_epochs=1, eval_data_loader=None):
        if self.train_engine is None:
            raise RuntimeError("Trainer engine does not exist")
        if self.eval_engine is not None:
            if eval_data_loader is not None:
                def _run_evaluator(engine):
                    self.eval_engine.run(eval_data_loader, max_epochs=1)

                if not self.train_engine.has_event_handler(_run_evaluator, Events.EPOCH_COMPLETED):
                    self.train_engine.add_event_handler(Events.EPOCH_COMPLETED, _run_evaluator)
                    # move handler to first position
                    event_handler = self.train_engine._event_handlers[Events.EPOCH_COMPLETED]
                    self.train_engine._event_handlers[Events.EPOCH_COMPLETED] = event_handler[-1:] + event_handler[0:-1]
            else:
                Warning("For model evaluation you have to provide eval_data_loader.")
        self.train_engine.run(train_data_loader, max_epochs=num_epochs)

    def early_stopping(self, metric, patience):
        # Early stopping with number of patience event when score_fn gives a result lower than the best result
        def _score_function(engine):
            return engine.state.metrics[metric]

        es_handler = EarlyStopping(patience=patience, score_function=_score_function, trainer=self.train_engine)
        self.train_engine.add_event_handler(Events.COMPLETED, es_handler)
        setup_logger("es_handler")

    def lr_scheduler(self, gamma=0.975):
        lr_scheduler = ExponentialLR(self.optimizer, gamma=gamma)
        self.train_engine.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: lr_scheduler.step())


def get_model_from_trainer(trainer):
    return {"net": trainer.model}
