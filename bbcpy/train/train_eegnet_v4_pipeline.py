"""
Created on 31.08.21
@author :ali
"""
import datetime
import gc
import pprint
import signal
import subprocess

import optuna
from ignite.engine import Events
from ignite.metrics import Accuracy, Loss, Precision, Recall, Fbeta, ConfusionMatrix
from sklearn import preprocessing

from bbcpy.models.eegnet import EEGNetv4
from bbcpy.train.bbcpy_logger import log_training, save_model_summary, log_pBar, save_args_to_yaml
from bbcpy.train.bbcpy_trainer import EEGBasedTrainer
from bbcpy.train.utils import tune_optimizer, tune_batch_size, tune_input_norm
from bbcpy.utils.argparser import yaml_argparse
from bbcpy.utils.data import *
from bbcpy.utils.file_management import *
from bbcpy.utils.subject import prepare_dataset, load_bbci_data
from bbcpy.utils.visualization import plot_cm


def _prepare_batch(batch, device=None, non_blocking=False):
    """Factory function to prepare batch for training: pass to a device with options
    """
    x, y = batch

    return (x.to(device=device, non_blocking=non_blocking),
            y.to(device=device, non_blocking=non_blocking).float())


global NUM_CLASSES


def run(baseline_mode, raw_args=None):
    pid = os.getpid()
    subprocess.Popen("renice -n 10 -p {}".format(pid), shell=True)

    # free gpu memory after finish or exception raised
    def _empty_cuda_cache(engine):
        torch.cuda.empty_cache()
        gc.collect()
        os.kill(pid, signal.SIGSTOP)

    ################################## Load arguments ##################################
    # Need to change here

    root_dir = get_dir_by_indicator(indicator="ROOT")
    baseline_yaml_path = str(
        Path(root_dir) / "bbcpy" / "params" / "{:}.yaml".format("baselines_eegnet_v4")) + "::" + baseline_mode

    args = yaml_argparse(baseline_yaml_path, raw_args=raw_args)

    if args.logging.log_dir == "":
        log_dir = Path(root_dir) / 'runs/{}/{}/{}'.format(
            args.experiment_name, args.run_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        root_path = os.path.join(Path(root_dir), "runs")
        create_working_folder(root_path=root_path, experiment_name=args.experiment_name, run_name=args.run_name)
        args.logging.log_dir = log_dir
    ################################## Machine Setup ##################################

    device = args.data.device
    if device == "cuda":
        torch.device("cuda")
    else:
        torch.device('cpu')

    ################################## Seed Setup  ##################################
    torch.random.manual_seed(args.data.seed)
    torch.cuda.manual_seed(args.data.seed)
    SEED = np.random.seed(args.tunning.seed)

    if args.data.task_type == "LR":
        NUM_CLASSES = 1

    INPUT_SHAPE = (62, int(args.data.time_interval[-1] - args.data.time_interval[0]))

    ################################## Create Model ##################################

    model = EEGNetv4(in_chans=INPUT_SHAPE[0],
                     n_classes=NUM_CLASSES,
                     input_window_samples=INPUT_SHAPE[1],
                     final_conv_length=args.network.final_conv_length,
                     pool_mode=args.network.pool_mode,
                     F1=args.network.F1,
                     D=args.network.D,
                     F2=args.network.F2,  # usually set to F1*D (?)
                     kernel_length=args.network.kernel_length,
                     third_kernel_size=tuple(args.network.third_kernel_size),
                     drop_prob=args.network.drop_prob, )

    model.to(device)

    ################################## Optimizer and Loss function ##################################

    optimizer = getattr(torch.optim, args.network.optim)(model.parameters(), lr=float(args.network.lr))
    loss_fn = getattr(torch.nn, args.network.loss)()

    ################################## Metrics ##################################

    def _thresholded_output_transform(output):
        y_pred, y, loss = output
        return y_pred.gt(0.5).double(), y.double()

    def _thresholded_output_transform_cm(output):
        y_pred, y, loss = output
        y_pred = y_pred.gt(0.5).cpu().to(torch.int)
        y = y.cpu().to(torch.int)
        # 1. INSTANTIATE
        enc = preprocessing.OneHotEncoder()
        # 2. FIT
        enc.fit(y)
        # 3. Transform
        onehot_pred = enc.transform(y_pred).toarray()
        return torch.from_numpy(onehot_pred), y.squeeze()

    METRICS = {
        args.network.loss: Loss(loss_fn, output_transform=_thresholded_output_transform),
        'accuracy': Accuracy(output_transform=_thresholded_output_transform),
        "precision": Precision(average=True, output_transform=_thresholded_output_transform),
        "recall": Recall(average=True, output_transform=_thresholded_output_transform),
        "F1": Fbeta(beta=1, average=True, output_transform=_thresholded_output_transform),
        'cm': ConfusionMatrix(num_classes=2, output_transform=_thresholded_output_transform_cm)}

    ################################## Create Trainer, Evaluator and Tester Engine #######################
    bbcpy_trainer = EEGBasedTrainer(model=model, optimizer=optimizer, loss=loss_fn, metrics=METRICS, args=args,
                                    log_dir=log_dir, device=device, batch_transform=_prepare_batch)

    ################################## Training & Tunning ######################################
    if args.pipeline_mode == "TRAINING":

        ################################## Load Training and Validation Set ##################################
        raw_data_train = load_bbci_data(args.data.data_path, args.data.subjects_list,
                                        args.data.sessions_list, args.data.task_type,
                                        args.data.time_interval, args.data.merge_sessions, args.data.reshape_type)

        data, train_loader, eval_loader, _, num_samples_train, data_norm_params = \
            prepare_dataset(raw_data_train, args.data.norm_type,
                            args.data.norm_axis, args.data.reshape_axes,
                            args.data.train_dev_test_split, args.data.batch_size)

        bbcpy_trainer.create_train_engine(metric_names=args.network.train_metrics_name,
                                          non_blocking=args.data.non_blocking)
        bbcpy_trainer.create_eval_engine(metric_names=args.network.eval_metrics_name,
                                         non_blocking=args.data.non_blocking)

        if args.network.lr_decay:
            bbcpy_trainer.lr_scheduler(gamma=args.network.gamma)

        if args.network.Earlystopping:
            bbcpy_trainer.early_stopping(metric=args.network.checkpoint_metric, patience=args.network.patience)

        save_model_summary(path=log_dir, model=model, input_shape=INPUT_SHAPE, batch_size=args.data.batch_size,
                           device=device)

        ################################## Tunning ######################################
        if args.tunning.enable_optuna_tuner:

            def objective(trial):

                opt_train_loader = train_loader
                opt_eval_loader = eval_loader

                model = EEGNetv4(in_chans=INPUT_SHAPE[0],
                                 n_classes=NUM_CLASSES,
                                 input_window_samples=INPUT_SHAPE[1],
                                 final_conv_length=args.network.final_conv_length, trial=trial)
                model.to(device)

                ################################## Tunning Training Hyperparameters##################################
                if isinstance(args.tunning.input_norm, list):
                    opt_train_loader, opt_eval_loader, args.data.norm_type = tune_input_norm(trial, raw_data_train,
                                                                                             args)

                if isinstance(args.tunning.batch_size, list):
                    opt_train_loader, opt_eval_loader, args.data.batch_size = tune_batch_size(trial, raw_data_train,
                                                                                              args)

                if isinstance(args.tunning.optimizers, list):
                    optimizers = tune_optimizer(trial, model, args.tunning.optimizers, args.tunning.lrs)
                    bbcpy_trainer.optimizer = optimizers

                if args.logging.enable_Summary_save:
                    save_model_summary(path=log_dir, model=model, input_shape=INPUT_SHAPE,
                                       batch_size=args.data.batch_size,
                                       device=device, trial_id=trial._trial_id - 1)

                bbcpy_trainer.model = model

                bbcpy_trainer.create_train_engine(metric_names=args.network.train_metrics_name,
                                                  non_blocking=args.data.non_blocking)
                bbcpy_trainer.create_eval_engine(metric_names=args.network.eval_metrics_name,
                                                 non_blocking=args.data.non_blocking)

                bbcpy_trainer.lr_scheduler(gamma=args.network.gamma)

                # Register a pruning handler to the evaluator.
                pruning_handler = optuna.integration.PyTorchIgnitePruningHandler(trial, "accuracy",
                                                                                 bbcpy_trainer.train_engine)

                bbcpy_trainer.eval_engine.add_event_handler(Events.COMPLETED, pruning_handler)

                bbcpy_trainer.update_trial(trial)

                log_training(bbcpy_trainer, model, args)

                bbcpy_trainer.run_trainer(opt_train_loader, num_epochs=args.data.num_epochs,
                                          eval_data_loader=opt_eval_loader)

                return bbcpy_trainer.eval_engine.state.metrics[args.tunning.metric]

            pruner = optuna.pruners.MedianPruner()

            storage = "sqlite:///{:}/study_{:}_{:}.db".format(log_dir, args.experiment_name, args.run_name)
            sampler = getattr(optuna.samplers, args.tunning.sampler)(seed=SEED)
            study = optuna.create_study(direction=args.tunning.direction,
                                        sampler=sampler,
                                        pruner=pruner,
                                        study_name="study_{:}_{:}".format(args.experiment_name, args.run_name),
                                        storage=storage,
                                        load_if_exists=False)

            study.optimize(objective, n_trials=args.tunning.n_trials, timeout=args.tunning.timeout)
        else:

            save_args_to_yaml(args, model)
            tb_logger = log_training(bbcpy_trainer, model, args)

            bbcpy_trainer.run_trainer(train_data_loader=train_loader, num_epochs=args.data.num_epochs,
                                      eval_data_loader=eval_loader)
            tb_logger.close()
        ################################## Terminate  ######################################
        bbcpy_trainer.train_engine.add_event_handler(Events.TERMINATE, _empty_cuda_cache)
        bbcpy_trainer.train_engine.add_event_handler(Events.EXCEPTION_RAISED, _empty_cuda_cache)
        bbcpy_trainer.eval_engine.add_event_handler(Events.TERMINATE, _empty_cuda_cache)
        bbcpy_trainer.eval_engine.add_event_handler(Events.EXCEPTION_RAISED, _empty_cuda_cache)

    # ************************************************ TESTING  **************************************#
    if args.pipeline_mode == "TESTING":
        bbcpy_trainer.create_eval_engine(name="test_engine", metric_names=args.network.test_metrics_name,
                                         non_blocking=args.data.non_blocking)

        raw_data_test = load_bbci_data(args.data.data_path, args.data.subjects_list,
                                       args.data.sessions_list, args.data.task_type,
                                       args.data.time_interval, args.data.merge_sessions, args.data.reshape_type)

        data, test_loader, _, _, num_samples_test, data_norm_params_test = \
            prepare_dataset(raw_data_test, args.data.norm_type,
                            args.data.norm_axis, args.data.reshape_axes, [1., .0, 0.], batch_size=0)

        test_metrics_name = ["accuracy", "precision", "recall", "F1"]
        file_name = os.path.join(log_dir, "Testing_summary.txt")

        def log_test_results():
            pbar = log_pBar(bbcpy_trainer.eval_engine, test_metrics_name)
            metrics = {kk: bbcpy_trainer.eval_engine.state.metrics[kk] for kk in test_metrics_name}

            with Tee(file_name):
                pbar.log_message(
                    "Validation Results - Epoch: {} \nMetrics\n{}"
                        .format(bbcpy_trainer.eval_engine.state.epoch, pprint.pformat(metrics)))
                pbar.n = pbar.last_print_n = 0

        bbcpy_trainer.eval_engine.add_event_handler(Events.COMPLETED, log_test_results)

        def confusion_matrix_fig():
            fig = plot_cm(engine=bbcpy_trainer.eval_engin, class_type=args.data.task_type)
            fig_path = os.path.join(log_dir, "cm_test.png")
            fig.savefig(fig_path, bbox_inches="tight")

        bbcpy_trainer.eval_engine.add_event_handler(Events.COMPLETED, confusion_matrix_fig)
        bbcpy_trainer.run_evaluator(data_loader=test_loader)
        ################################## Terminate  ######################################
        bbcpy_trainer.eval_engine.add_event_handler(Events.TERMINATE, _empty_cuda_cache)
        bbcpy_trainer.eval_engine.add_event_handler(Events.EXCEPTION_RAISED, _empty_cuda_cache)


if __name__ == "__main__":
    run(baseline_mode="debug_pipeline", raw_args=None)
    # run(baseline_mode="tune_pipeline", raw_args=None)
    # # run(baseline_mode="train_pipeline", raw_args=None)
    # run(baseline_mode="test_pipeline", raw_args=None)
