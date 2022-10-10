"""
Created on 29.08.21
@author :ali
"""
import argparse

from ignite.engine import Events
from torch import optim

from bbcpy.utils.subject import prepare_dataset



def _collect_metric_on_epoch(engine, metrics_engine, metric, ep, metrics_dict):
    if engine.state.epoch == ep:
        key = "{:}{:}".format(metric, ep)
        if isinstance(metrics_engine, list):
            num_metrics = len(metrics_engine)
            value = 0.
            for name, engine, _ in metrics_engine:
                value += engine.state.metrics[metric]
            metrics_dict[key] = value / num_metrics
        else:
            metrics_dict[key] = metrics_engine.state.metrics[metric]


def log_metrics_on_epochs(engine, metrics_engine, metrics, epochs):
    metrics_dict = {}
    for metric in metrics:
        for epoch in epochs:
            engine.add_event_handler(Events.EPOCH_COMPLETED, _collect_metric_on_epoch, metrics_engine, metric,
                                     epoch, metrics_dict)
    return metrics_dict


def args_to_dict(args):
    """Creates a dict from (nested) argparse.Namespace objects

    :param args: parsed args
    :type args: argparse.Namespace
    :returns: args as dict
    :rtype: dict
    """

    d = vars(args).copy()
    for k, v in d.items():
        if isinstance(v, argparse.Namespace):
            d[k] = args_to_dict(v)

    return d


def tune_batch_size(trial, raw_data, args):
    batch_size = trial.suggest_categorical("batch_size", args.tunning.batch_size)
    data, train_loader_optuna, eval_loader_optuna, _, num_samples_train, data_norm_params = \
        prepare_dataset(raw_data, args.data.norm_type,
                        args.data.norm_axis, args.data.reshape_axes,
                        args.data.train_dev_test_split, batch_size)

    return train_loader_optuna, eval_loader_optuna, batch_size


def tune_optimizer(trial, model, optuna_optimizers, optuna_lrs):
    optimizer_name = trial.suggest_categorical("optimizer", optuna_optimizers)
    lr = trial.suggest_uniform("lr", optuna_lrs[0], optuna_lrs[1])

    return getattr(optim, optimizer_name)(model.parameters(), lr=lr)

def tune_input_norm(trial, raw_data, args):
    norm_type = trial.suggest_categorical("input_norm", args.tunning.input_norm)
    norm_axes = trial.suggest_int("norm_axes", low=0, high=2, step=1)
    data, train_loader_optuna, eval_loader_optuna, _, num_samples_train, data_norm_params = \
        prepare_dataset(raw_data, norm_type,
                        norm_axes, args.data.reshape_axes,
                        args.data.train_dev_test_split, args.data.batch_size)

    return train_loader_optuna, eval_loader_optuna, norm_type