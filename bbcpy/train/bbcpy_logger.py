"""
Created on 29.08.21
@author :ali
"""
import os

import numpy as np
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint
from tensorboardX.summary import hparams
from torchsummary import summary

from bbcpy.train.bbcpy_trainer import get_model_from_trainer
from bbcpy.train.utils import log_metrics_on_epochs
from bbcpy.utils.argparser import args_to_dict, args_to_yaml
from bbcpy.utils.file_management import Tee
from bbcpy.utils.visualization import plot_cm

train_metrics = ["accuracy", "precision", "recall", "F1"]
eval_metrics = ["accuracy", "precision", "recall", "F1"]


def log_hparams(tensorboard_logger: TensorboardLogger, train_engine, metrics_dict,
                param_dict):
    param_dict = dict(param_dict)
    rem_keys = []

    for k, v in param_dict.items():

        if type(v) == bool:
            v = int(v)
        if type(v) == list or type(v) == tuple:
            if all([isinstance(e, str) for e in v]):
                v = "-".join(v)
            elif any([isinstance(e, str) for e in v]):
                v = "-".join([str(e) for e in v])
            elif all([isinstance(e, bool) for e in v]):
                v = "-".join(str(int(e)) for e in v)
            elif any([isinstance(e, int) for e in v]):
                v = "-".join(str(i) for i in v)
        elif v is None:
            rem_keys.append(k)
        param_dict[k] = v

    for k in rem_keys:
        del param_dict[k]

    writer = tensorboard_logger.writer

    def _add_hparams(engine, *args, **kwargs):
        add_hparams_to_writer(writer, param_dict, metrics_dict)

    train_engine.add_event_handler(Events.COMPLETED, _add_hparams)


def add_hparams_to_writer(writer, param_dict=None, metrics_dict=None):
    exp, ssi, sei = hparams(param_dict, metrics_dict)
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)
    for k, v in metrics_dict.items():
        writer.add_scalar(k, v)


def log_model(tensorboard_logger: TensorboardLogger, train_engine, model, tag=None,
              event=Events.EPOCH_COMPLETED,
              log_grads_scalar=False,
              log_grads_hist=True,
              log_weights_scalar=False,
              log_weights_hist=True):
    """Logs a model to tensorboard
    """

    if log_grads_scalar:
        tensorboard_logger.attach(train_engine, GradsScalarHandler(model, reduction=torch.norm, tag=tag),
                                  event_name=event)
    if log_grads_hist:
        tensorboard_logger.attach(train_engine, GradsHistHandler(model, tag=tag), event_name=event)

    if log_weights_scalar:
        tensorboard_logger.attach(train_engine, WeightsScalarHandler(model, reduction=torch.norm, tag=tag),
                                  event_name=event)
    if log_weights_hist:
        tensorboard_logger.attach(train_engine, WeightsHistHandler(model, tag=tag), event_name=event)


def tensorbaord_logging(trainer, tensorboard_logger, args, model, hparam_dict,
                        log_grads_scalar=True,
                        log_grads_hist=True,
                        log_weights_scalar=False,
                        log_weights_hist=False):
    if args.logging.enable_tensorboard_logger:

        model_dict = get_model_from_trainer(trainer)

        for k, v in model_dict.items():
            log_model(tensorboard_logger,
                      trainer.train_engine,
                      v,
                      tag=k,
                      event=Events.EPOCH_COMPLETED,
                      log_grads_hist=log_grads_hist,
                      log_grads_scalar=log_grads_scalar,
                      log_weights_hist=log_weights_hist,
                      log_weights_scalar=log_weights_scalar)

        tensorboard_logger.attach(trainer.train_engine,
                                  OutputHandler("0_Train", metric_names=train_metrics),
                                  event_name=Events.EPOCH_COMPLETED)

        def _global_step_transform(*args, **kwargs):
            return trainer.train_engine.state.epoch

        if eval_metrics is not None:
            tensorboard_logger.attach(trainer.eval_engine,
                                      OutputHandler("1_Eval",
                                                    metric_names=eval_metrics,
                                                    global_step_transform=_global_step_transform),
                                      event_name=Events.EPOCH_COMPLETED)

        if args.logging.enable_model_graph:
            data_points = args.data.time_interval[-1] - args.data.time_interval[0]
            tensorboard_logger.writer.add_graph(model, torch.zeros((args.data.batch_size, 62, data_points)).to(args.data.device))

        if args.logging.enable_Hparam_tensorboard_logger:
            epochs = tuple(np.floor(np.linspace(0, args.data.num_epochs, 100)[1::]).astype(np.int))
            hparam_metrics = log_metrics_on_epochs(trainer.train_engine,
                                                   trainer.eval_engine,
                                                   (args.network.loss,),
                                                   epochs)
            log_hparams(tensorboard_logger, trainer.train_engine, hparam_metrics, hparam_dict)

        trainer.train_engine.add_event_handler(Events.COMPLETED, lambda x: tensorboard_logger.close())
    else:
        tensorboard_logger = None


def log_training(trainer, model, args,
                 log_grads_scalar=False,
                 log_grads_hist=True,
                 log_weights_scalar=False,
                 log_weights_hist=True):
    """Logs the training process to tensorboard and console
    """

    score_sign = 1.

    hparam_dict = args_to_dict(args)
    hparam_dict["num_parameters"] = model.parameters()
    log_dir = trainer.log_dir

    if args.tunning.enable_optuna_tuner:
        trial_id = trainer.trial._trial_id
        tensorboard_logger = TensorboardLogger(
            log_dir="{:}/{:}/trial_{:}".format(log_dir, "tensorboard", str(trial_id - 1)))
        tensorbaord_logging(trainer, tensorboard_logger, args, model, hparam_dict,
                            log_grads_hist=log_grads_hist,
                            log_grads_scalar=log_grads_scalar,
                            log_weights_hist=log_weights_hist,
                            log_weights_scalar=log_weights_scalar)

        if args.logging.enable_model_checkpointing:
            model_checkpoint = ModelCheckpoint(
                "{:}/{:}".format(log_dir, "checkpoints"),
                n_saved=1,
                filename_prefix="best",
                score_function=lambda x: score_sign * trainer.eval_engine.state.metrics[args.network.checkpoint_metric],
                score_name="trial_{:}_validation_acc".format(trial_id - 1),
                global_step_transform=global_step_from_engine(trainer.eval_engine),
                require_empty=False,
            )
            trainer.eval_engine.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

    else:
        tensorboard_logger = TensorboardLogger(log_dir="{:}/{:}".format(log_dir, "tensorboard"))
        tensorbaord_logging(trainer, tensorboard_logger, args, model, hparam_dict,
                            log_grads_hist=True,
                            log_grads_scalar=log_grads_scalar,
                            log_weights_hist=True,
                            log_weights_scalar=log_weights_scalar)
        model_checkpoint = ModelCheckpoint(
            "{:}/{:}".format(log_dir, "checkpoints"),
            n_saved=1,
            filename_prefix="best",
            score_function=lambda x: score_sign * trainer.eval_engine.state.metrics[args.network.checkpoint_metric],
            score_name="validation_{:}".format(args.network.checkpoint_metric),
            global_step_transform=global_step_from_engine(trainer.train_engine),
            require_empty=False,
        )
        trainer.eval_engine.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

        if args.logging.plot_confusion_matrix:
            def confusion_matrix_fig():
                fig = plot_cm(engine=trainer.eval_engine, class_type=args.data.task_type)
                tensorboard_logger.writer.add_figure(tag="Confusion Matrix", figure=fig,
                                                     global_step=trainer.train_engine.state.epoch)

            trainer.eval_engine.add_event_handler(Events.EPOCH_COMPLETED(every=args.logging.save_interval),
                                                  confusion_matrix_fig)

        # Log optimizer parameters
        if args.logging.log_optimizer_lr:
            tensorboard_logger.attach(trainer.train_engine,
                                      log_handler=OptimizerParamsHandler(trainer.optimizer, "lr"),
                                      event_name=Events.EPOCH_STARTED)

    if args.logging.enable_tabular_logger:
        log_tabular(trainer)




    return tensorboard_logger


def log_tabular(trainer):
    tab_logger = ProgressBar(persist=True)
    tab_logger.attach(trainer.train_engine, metric_names=train_metrics)
    # tab_logger.attach(trainer.eval_engine, metric_names=train_metrics)


def log_pBar(engine, metrics_name):
    pbar = ProgressBar(persist=True, bar_format="")
    pbar.attach(engine, metrics_name)
    return pbar


def save_model_summary(path, model, input_shape, batch_size, device, trial_id=None):
    wdir = os.path.join(path, "model_summary")
    if trial_id is not None:
        file_name = os.path.join(wdir, "model_summary_trial_{:}.txt".format(trial_id))
    else:
        os.makedirs(wdir, exist_ok=True)
        file_name = os.path.join(wdir, "model_summary.txt")
    with Tee(file_name):
        summary(model, input_shape, batch_size, device)


def save_args_to_yaml(args, model):
    hparam_dict = args_to_dict(args)
    hparam_dict["num_parameters"] = model.parameters()
    args_to_yaml(args, os.path.join(args.logging.log_dir, "Hparams.yaml"))
