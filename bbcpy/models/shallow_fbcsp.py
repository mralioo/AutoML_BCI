import optuna
from torch import nn
from torch.nn import init
from torchsummary import summary
import numpy as np

from bbcpy.models.utils.functions import *
from bbcpy.models.utils.modules import Expression, Ensure4d
from bbcpy.models.utils.util import np_to_var


class ShallowFBCSPNet(nn.Sequential):
    """Shallow ConvNet model from [2]_.

    Parameters
    ----------
    in_chans : int
        XXX

    References
    ----------
    .. [2] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
    """

    def __init__(
            self,
            in_chans,
            n_classes,
            input_window_samples=None,
            n_filters_time=40,
            filter_time_length=25,
            n_filters_spat=40,
            pool_time_length=75,
            pool_time_stride=15,
            final_conv_length=30,
            conv_nonlin=square,
            pool_mode="mean",
            pool_nonlin=safe_log,
            split_first_layer=False,
            batch_norm=True,
            batch_norm_alpha=0.1,
            drop_prob=0.5,
            trial=None
    ):

        if isinstance(trial, optuna.Trial):
            self.n_filters_time = trial.suggest_int("n_filters_time", low=20, high=60, step=10)
            self.filter_time_length = trial.suggest_int("filter_time_length", low=60, high=100, step=10)
            self.n_filters_spat = trial.suggest_int("n_filters_spat", low=20, high=60, step=10)
            self.pool_time_length = trial.suggest_int("pool_time_length", low=10, high=50, step=10)
            self.pool_time_stride = trial.suggest_int("pool_time_stride", low=15, high=45, step=15)
            self.drop_prob = trial.suggest_float("drop_prob", 0, 1)
            self.batch_norm = trial.suggest_categorical("batch_norm", [True, False])
            self.batch_norm_alpha = trial.suggest_float("batch_norm_alpha", 0, 1)


        else:
            self.n_filters_time = n_filters_time
            self.filter_time_length = filter_time_length
            self.n_filters_spat = n_filters_spat
            self.pool_time_length = pool_time_length
            self.pool_time_stride = pool_time_stride
            self.drop_prob = drop_prob
            self.batch_norm = batch_norm
            self.batch_norm_alpha = batch_norm_alpha

        super().__init__()
        if final_conv_length == "auto":
            assert input_window_samples is not None
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.input_window_samples = input_window_samples
        self.final_conv_length = final_conv_length
        self.split_first_layer = split_first_layer
        self.conv_nonlin = conv_nonlin
        self.pool_mode = pool_mode
        self.pool_nonlin = pool_nonlin

        self.add_module("ensuredims", Ensure4d())
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        if self.split_first_layer:
            self.add_module("dimshuffle", Expression(transpose_time_to_spat))
            self.add_module(
                "conv_time",
                nn.Conv2d(
                    1,
                    self.n_filters_time,
                    (self.filter_time_length, 1),
                    stride=1,
                ),
            )
            self.add_module(
                "conv_spat",
                nn.Conv2d(
                    self.n_filters_time,
                    self.n_filters_spat,
                    (1, self.in_chans),
                    stride=1,
                    bias=not self.batch_norm,
                ),
            )
            n_filters_conv = self.n_filters_spat
        else:
            self.add_module(
                "conv_time",
                nn.Conv2d(
                    self.in_chans,
                    self.n_filters_time,
                    (self.filter_time_length, 1),
                    stride=1,
                    bias=not self.batch_norm,
                ),
            )
            n_filters_conv = self.n_filters_time
        if self.batch_norm:
            self.add_module(
                "bnorm",
                nn.BatchNorm2d(
                    n_filters_conv, momentum=self.batch_norm_alpha, affine=True
                ),
            )
        self.add_module("conv_nonlin_exp", Expression(self.conv_nonlin))
        self.add_module(
            "pool",
            pool_class(
                kernel_size=(self.pool_time_length, 1),
                stride=(self.pool_time_stride, 1),
            ),
        )
        self.add_module("pool_nonlin_exp", Expression(self.pool_nonlin))
        self.add_module("drop", nn.Dropout(p=self.drop_prob))
        self.eval()
        if self.final_conv_length == "auto":
            out = self(
                np_to_var(
                    np.ones(
                        (1, self.in_chans, self.input_window_samples, 1),
                        dtype=np.float32,
                    )
                )
            )
            n_out_time = out.cpu().data.numpy().shape[2]
            self.final_conv_length = n_out_time
        self.add_module(
            "conv_classifier",
            nn.Conv2d(
                n_filters_conv,
                self.n_classes,
                (self.final_conv_length, 1),
                bias=True, stride=1
            ),
        )

        self.add_module("sigmoid", nn.Sigmoid())
        self.add_module("squeeze", Expression(squeeze_final_output))
        # self.add_module("squeeze", Expression(squeeze_output))

        # need to check the last layer output
        # self.add_module("fc_1", nn.Linear(42, 12))
        #
        # self.add_module("squeeze_fc", Expression(squeeze_dim))
        # self.add_module("fc_2", nn.Linear(12, 1))


        # Initialization, xavier is same as in paper...
        init.xavier_uniform_(self.conv_time.weight, gain=1)
        # maybe no bias in case of no split layer and batch norm
        if self.split_first_layer or (not self.batch_norm):
            init.constant_(self.conv_time.bias, 0)
        if self.split_first_layer:
            init.xavier_uniform_(self.conv_spat.weight, gain=1)
            if not self.batch_norm:
                init.constant_(self.conv_spat.bias, 0)
        if self.batch_norm:
            init.constant_(self.bnorm.weight, 1)
            init.constant_(self.bnorm.bias, 0)
        init.xavier_uniform_(self.conv_classifier.weight, gain=1)
        init.constant_(self.conv_classifier.bias, 0)


if __name__ == "__main__":
    from torch.autograd import Variable
    import numpy as np

    # x = Variable(torch.from_numpy(np.random.randn(1, 44, 534)))
    # x = Variable(torch.from_numpy(np.random.randn(1, 62, 6000)))
    x = Variable(torch.zeros((1, 62, 6000)))
    model = ShallowFBCSPNet(in_chans=62, n_classes=1, input_window_samples=6000,
                            n_filters_time=32,
                            filter_time_length=80,
                            n_filters_spat=32,
                            pool_time_length=30,
                            pool_time_stride=30,
                            final_conv_length="auto",
                            conv_nonlin=square,
                            pool_mode="mean",
                            pool_nonlin=safe_log,
                            split_first_layer=False,
                            batch_norm=False,
                            batch_norm_alpha=0.1,
                            drop_prob=0.5,
                            )
    summary(model, (62, 6000), device="cpu")
    # s = get_output_shape(model, 62, 6000)
    # y_pred = model(x)
