import os
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset, BatchSampler, SubsetRandomSampler, SequentialSampler, DataLoader

from bbcpy.utils.file_management import get_dir_by_indicator


def load_matlab_data(subjectno, sessionno, data_path, outfile=False):
    '''
    Usage:
        data, fs, clab, mnt, mrk_class, mrk_className, task_type, task_typeName, timepoints = load_matlab_data(subjectno, sessionno)
        load_matlab_data(subjectno, sessionno, outfile)
    Parameters:
        subjectno: Number of the subject
        sessionno: Number of the session
        data_path: data relative path
        outfile: optional, needed if the data is to be saved to an npz file, name of the output file
    Returns:
        data: a 3D array of epochs of multi-channel timeseries (trials x channels x samples), unit [uV] 
        fs:   sampling frequency [Hz]
        clab: a 1D array of channel names  (channels)
        mnt:  a 2D array of channel coordinates (channels x 2)   #or rather, it will be, eventually, right now it's 3D coordinates
              The electrode montage "mnt" holds the information of the 
              2D projected positions of the channels, i.e. electrodes, 
              on the scalp - seem from the top with nose up.
        mrk_class: a 1D array that assigns markers to classes (0-3)
        mrk_className: a list that assigns class names to classes
        task_type: a 1D array describing which task was given (0-2)
        task_typeName: a list that assigns names to the task type ('LR', 'UD', '2D')
        timepoints = a 2D array of the trial time (in ms) relative to target presentation, varying in length (trials x timepoints)
    '''

    if data_path is None:
        DATA_PATH = Path(
            os.path.abspath('')).parent.parent.parent / "data"  # the data folder has to be in the parent folder!
    else:
        DATA_PATH = Path(data_path)
    mat_files = []
    pattern = "*.mat"
    for file in DATA_PATH.glob(pattern):
        mat_files.append(file)

    group_files = {}
    for f in mat_files:
        res = f.stem.split("_", 1)
        if res[0] in group_files:
            group_files[res[0]][res[1]] = f
        else:
            group_files[res[0]] = {}
            group_files[res[0]][res[1]] = f

    # Select the subject person and the session number to load the data 
    subject = "S" + str(subjectno)
    session = "Session_" + str(sessionno)
    mdata = loadmat(os.path.join(DATA_PATH, group_files[subject][session]), mat_dtype=True)["BCI"]
    ndata = {n: mdata[n][0, 0] for n in mdata.dtype.names}

    # data and fs can be used as is, the rest needs further extraction
    data = ndata["data"].squeeze()
    fs = ndata["SRATE"].squeeze()
    time = ndata["time"].squeeze()
    chaninfo = ndata["chaninfo"].squeeze()
    trialdata = ndata["TrialData"].squeeze()

    # Extracting 2D array with timepoints (trials x timepoints)
    timepoints = []
    for t in time:
        timepoints.append(t[0][:])
    timepoints = np.array(timepoints)

    # Extracting the channel names
    clab = []
    for label in chaninfo['label'].flatten()[0][0]:
        # print (label[0])
        clab.append(label[0])
    clab = np.array(clab)

    # Extracting the electrode montage, excluding the montage of the reference electrode
    mnt = []
    for position in chaninfo['electrodes'].flatten()[0][0][:-1]:
        mnt.append([position[1][0][0], position[2][0][0], position[3][0][0]])
    mnt = np.array(mnt)

    # Extracting marker classes, original range 1-4, modified range 0-3 to match the indices of mrk_className
    # also extracting task type, original range 1-3, modified range 0-2 to match the indices of task_typeName
    mrk_className = ['right', 'left', 'up', 'down']
    task_typeName = ['LR', 'UD', '2D']
    mrk_class = []
    task_type = []
    for trial in trialdata.flatten():
        mrk_class.append(trial[3][0][0] - 1)
        task_type.append(trial[0][0][0] - 1)
    mrk_class = np.array(mrk_class)
    task_type = np.array(task_type)

    if (outfile):
        np.savez(outfile, data=data, fs=fs, clab=clab, mnt=mnt, mrk_class=mrk_class, mrk_className=mrk_className,
                 task_type=task_type, task_typeName=task_typeName, timepoints=timepoints)
        return
    else:
        return data, fs, clab, mnt, mrk_class, mrk_className, task_type, task_typeName, timepoints


def load_matlab_data_fast(subjectno, sessionno, data_path, outfile=False):
    if data_path is None:
        root_dir = get_dir_by_indicator(indicator="ROOT")
        DATA_PATH = Path(root_dir).parent / "data"  # the data folder has to be in the parent folder!
    else:
        DATA_PATH = Path(data_path)

    mat_files = []
    pattern = "*.mat"
    for file in DATA_PATH.glob(pattern):
        mat_files.append(file)

    group_files = {}
    for f in mat_files:
        res = f.stem.split("_", 1)
        if res[0] in group_files:
            group_files[res[0]][res[1]] = f
        else:
            group_files[res[0]] = {}
            group_files[res[0]][res[1]] = f

    # Select the subject person and the session number to load the data 
    subject = "S" + str(subjectno)
    session = "Session_" + str(sessionno)
    mdata = loadmat(os.path.join(DATA_PATH, group_files[subject][session]), mat_dtype=True)["BCI"]

    # data for all 450 trials:
    data = mdata[0][0][0][:][0]

    # time:
    timepoints = mdata[0][0][1][0]

    # srate
    fs = mdata[0][0][4][0]

    # Extracting channel names and mnt info, excluding the reference electrode
    chan_inf = np.array(mdata[0][0][7][0][0][2][0])
    clab = []
    mnt = []
    for element in chan_inf[:-1]:
        clab.append(str(element[0][0]))
        mnt.append([element[1][0][0], element[2][0][0], element[3][0][0]])
    mnt = np.array(mnt)
    clab = np.array(clab)

    # Extracting marker classes, original range 1-4, modified range 0-3 to match the indices of mrk_className
    # also extracting task type, original range 1-3, modified range 0-2 to match the indices of task_typeName
    # trial artifact from the original data set, if bandpass filtered data from any electrode crosses threshold of +-100 μV = True, else False
    mrk_className = ['right', 'left', 'up', 'down']
    task_typeName = ['LR', 'UD', '2D']
    meta = mdata[0][0][5][:][0]
    task_type = []
    mrk_class = []
    trial_artifact = []
    for i in range(0, len(meta)):
        task_type.append(meta[i][0][0][0] - 1)
        mrk_class.append(meta[i][3][0][0] - 1)
        trial_artifact.append(meta[i][9][0][0])
    mrk_class = np.array(mrk_class)
    task_type = np.array(task_type)
    trial_artifact = np.array(trial_artifact)

    if (outfile):
        np.savez(outfile, data=data, fs=fs, clab=clab, mnt=mnt, mrk_class=mrk_class, mrk_className=mrk_className,
                 task_type=task_type, task_typeName=task_typeName, timepoints=timepoints)
        return

    # trial artifact from the original data set, if bandpass filtered data from any electrode crosses threshold of +-100 μV = True, else False
    mrk_className = ['right', 'left', 'up', 'down']
    task_typeName = ['LR', 'UD', '2D']
    meta = mdata[0][0][5][:][0]
    task_type = []
    mrk_class = []
    trial_artifact = []
    for i in range(0, len(meta)):
        task_type.append(meta[i][0][0][0] - 1)
        mrk_class.append(meta[i][3][0][0] - 1)
        trial_artifact.append(meta[i][9][0][0])
    mrk_class = np.array(mrk_class)
    task_type = np.array(task_type)
    trial_artifact = np.array(trial_artifact)
    if (outfile):
        np.savez(outfile, data=data, fs=fs, clab=clab, mnt=mnt, mrk_class=mrk_class, mrk_className=mrk_className,
                 task_type=task_type, task_typeName=task_typeName, timepoints=timepoints, trial_artifact=trial_artifact)
        return

    else:
        return data, fs, clab, mnt, mrk_class, mrk_className, task_type, task_typeName, timepoints, trial_artifact


def normalize(data, norm_type="std", axis=None, keepdims=True, eps=10 ** -5, norm_params=None):
    if norm_params is not None:
        if norm_params["norm_type"] == "std":
            data_norm = (data - norm_params["mean"]) / norm_params["std"]
        elif norm_params["norm_type"] == "minmax":
            data_norm = (data - norm_params["min"]) / (norm_params["max"] - norm_params["min"])
        else:
            raise RuntimeError("norm type {:} does not exist".format(norm_params["norm_type"]))

    else:
        if norm_type == "std":
            data_std = data.std(axis=axis, keepdims=keepdims)
            data_std[data_std < eps] = eps
            data_mean = data.mean(axis=axis, keepdims=keepdims)
            data_norm = (data - data_mean) / data_std
            norm_params = dict(mean=data_mean, std=data_std, norm_type=norm_type, axis=axis, keepdims=keepdims)
        elif norm_type == "minmax":
            data_min = data.min(axis=axis, keepdims=keepdims)
            data_max = data.max(axis=axis, keepdims=keepdims)
            data_max[data_max == data_min] = data_max[data_max == data_min] + eps
            data_norm = (data - data_min) / (data_max - data_min)
            norm_params = dict(min=data_min, max=data_max, norm_type=norm_type, axis=axis, keepdims=keepdims)
        elif norm_type is None:
            data_norm, norm_params = data, None
        else:
            data_norm, norm_params = None, None
            ValueError("Only 'std' and 'minmax' are supported")
    return data_norm, norm_params


def unnormalize(data_norm, norm_params):
    if norm_params["norm_type"] == "std":
        data = data_norm * norm_params["std"] + norm_params["mean"]
    elif norm_params["norm_type"] == "minmax":
        data = data_norm * (norm_params["max"] - norm_params["min"]) + norm_params["min"]
    return data


class Data(Dataset):
    def __init__(self, inputs, targets, transform=None):
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.Tensor(inputs)
        if not isinstance(targets, torch.Tensor):
            targets = torch.Tensor(targets)

        assert inputs.shape[0] == targets.shape[0]

        self.inputs = inputs
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        sample = [self.inputs[idx, :], self.targets[idx, :]]
        if self.transform:
            sample = self.transform(sample)
        return sample[0], sample[1]


class GpuDataLoader:
    def __init__(self, data, sampler, batch_size=1024):
        self._data = data
        self._sampler = sampler
        self._batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
        self.sample_iter = iter(self._batch_sampler)

    def __next__(self):
        indices = next(self.sample_iter)  # may raise StopIteration
        batch = self._data[indices]
        return batch

    def __len__(self):
        return len(self._batch_sampler)

    def __iter__(self):
        self.sample_iter = iter(self._batch_sampler)
        return self


def create_data_loaders(data, train_dev_test_split=[0.8, 0.2, 0.], batch_size=1024, pin_memory=False, num_workers=0):
    indices = list(range(len(data)))
    np.random.shuffle(indices)

    if len(train_dev_test_split) == 2:
        train_dev_test_split = train_dev_test_split + [0.0]

    split_train_dev = int(np.round(train_dev_test_split[0] * len(data)))
    split_dev_test = split_train_dev + int(np.round(train_dev_test_split[1] * len(data)))

    train_indices = indices[:split_train_dev]
    dev_indices = indices[split_train_dev:split_dev_test]
    test_indices = indices[split_dev_test:]

    if batch_size == 0:
        batch_size = len(train_indices)
    else:
        batch_size = min(batch_size, len(train_indices))

    train_sampler = SubsetRandomSampler(train_indices)
    dev_sampler = SequentialSampler(dev_indices)
    test_sampler = SequentialSampler(test_indices)


    train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers,
                              pin_memory=pin_memory)
    dev_loader = DataLoader(data, batch_size=batch_size, sampler=dev_sampler, num_workers=num_workers,
                            pin_memory=pin_memory)

    if len(test_indices) > 0:
        test_loader = DataLoader(data, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers,
                                 pin_memory=pin_memory)
    else:
        test_loader = []

    return train_loader, dev_loader, test_loader, [len(train_indices), len(dev_indices), len(test_indices)]
