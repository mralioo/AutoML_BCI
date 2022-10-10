from torch.utils.data import TensorDataset

from bbcpy.utils.data import *
from bbcpy.utils.data import create_data_loaders


class Subject:
    # information we have: what the indicated target was
    # information we don't have (yet): which target was reached
    def __init__(self, data_path, number, sessionno, task_type=None, time_interval=None, merge_sessions=False,
                 reshape_type="left_pad"):
        # session independent
        self.data_path = data_path
        self.number = number
        self.sessionno = sessionno
        self.value = task_type
        if isinstance(time_interval, list):
            self.t_min = time_interval[0]
            self.t_max = time_interval[1]
        else:
            self.t_min = 0
            self.t_max = time_interval
        self.merge_session = merge_sessions
        self.reshape_type = reshape_type
        self.fs = 1000
        self.clab = []
        self.descriptions = {
            'mrk_className': [],
            'task_typeName': []
        }
        # session dependent, trial independent: mnt
        # session dependent, trial dependent: data, mrk_class, task_type, timepoints
        # list of dictionaries, one dictionary per session
        self.loaded_sessions = []
        self.bci_recording = []
        self.mnt = []
        self.mrk_class = []
        self.task_type = []
        self.timepoints = []
        if self.sessionno is not None:
            if isinstance(self.sessionno, list):
                print("Subject: {}".format(self.number))
                for s in self.sessionno:
                    try:
                        self.load_session(s)
                    except (FileNotFoundError, KeyError, OSError):
                        print("Error in subject " + str(self.number) + " session " + str(s))
            elif isinstance(self.sessionno, int):
                self.load_session(self.sessionno)
        if self.value is not None:
            self.select_task_class("task_type", self.value)
            self.clear_all_sessions(type="loaded")
        if self.t_min is not None and self.t_max is not None:
            self.reshape_trial(self.t_min, self.t_max, self.reshape_type)
            self.clear_all_sessions(type="filtered")
        if self.merge_session:
            self.merge_all_sessions()
            self.clear_all_sessions(type="all")

    def load_session(self, sessionno):
        # checking if the session has been loaded already
        for se in self.loaded_sessions:
            if (se['se_number'] == sessionno):
                print("Session %s has already been loaded." % sessionno)
                return
        # get all the variables
        data, fs, clab, mnt, mrk_class, mrk_className, task_type, task_typeName, timepoints, trial_artifact = load_matlab_data_fast(
            self.number, sessionno, data_path=self.data_path)
        if (not self.loaded_sessions):
            # one-time initialisation so it doesn't have to be stored multiple times
            self.fs = fs
            self.clab = clab
            self.descriptions['mrk_className'] = mrk_className
            self.descriptions['task_typeName'] = task_typeName
        session = {
            "se_number": sessionno,
            "data": data,
            "mnt": mnt,
            "mrk_class": mrk_class,
            "task_type": task_type,
            "trial_artifact": trial_artifact,
            "timepoints": timepoints,
        }
        self.loaded_sessions.append(session)
        print("Loaded session %s" % self.loaded_sessions[-1]['se_number'])

    def select_task_class(self, var, value):
        # rn the Subject can only have one kind of filtered sessions (either LR, UD or 2D), and all loaded sessions will be filtered
        self.filtered_sessions = []
        try:
            for se in self.loaded_sessions:
                # using the index of the description to figure out the numerical value of the task_type
                marker = self.descriptions[var + 'Name'].index(value)
                idx = np.where(se[var] == marker)
                filtered = {
                    "filtered_by": value,
                    "se_number": se['se_number'],
                    "data": se["data"][idx],
                    "mnt": se['mnt'],
                    "mrk_class": se["mrk_class"][idx],
                    "task_type": se["task_type"][idx],
                    "trial_artifact": se["trial_artifact"][idx],
                    "timepoints": se["timepoints"][idx],
                }
                self.filtered_sessions.append(filtered)
        except ValueError:
            print("No element %s found in %s" % (value, var))
        except KeyError:
            print("No argument called %s found" % var)
        except:
            print("Something unexpected went wrong, are all the variable names correct?")

    def reshape_trial(self, t_min, t_max, reshape_type):
        """
        Args:
            t_min: start timepoint
            t_max: end timepoint
        Returns:
        """
        self.reshaped_sessions = []
        if reshape_type == "left_pad":
            def _pad_zero_to_left(trial, t_min, t_max):
                length = abs(t_max - t_min)
                if trial.shape[-1] <= length:
                    delta = length - trial.shape[-1]
                    trial = np.pad(trial, ((0, 0), (delta + 1, 0)), "constant")
                    return trial[:, t_min: t_max]
                else:
                    # return np.flip(trial[:, t_min: t_max], axis=1)
                    return trial[:, t_min: t_max]

            for se in self.filtered_sessions:
                filtered = {
                    "time_filter": (t_min, t_max),
                    "se_number": se["se_number"],
                    "data": np.array([_pad_zero_to_left(trial, t_min, t_max) for trial in se["data"]]),
                    "mnt": se["mnt"],
                    "mrk_class": se["mrk_class"],
                    "task_type": se["task_type"],
                    "timepoints": np.arange(t_min, t_max - 1, 1),
                }
                self.reshaped_sessions.append(filtered)
        if reshape_type == "slice":
            for se in self.filtered_sessions:
                idx = [i for i in range(len(se["data"])) if se["data"][i].shape[-1] >= int(t_max)]
                filtered = {
                    "time_filter": (t_min, t_max),
                    "se_number": se["se_number"],
                    "data": np.array([se["data"][i][:, t_min:t_max] for i in idx]),
                    "mnt": se["mnt"],
                    "mrk_class": se["mrk_class"][idx],
                    "task_type": se["task_type"][idx],
                    "timepoints": np.arange(t_min, t_max, 1),
                }
                self.reshaped_sessions.append(filtered)

    def merge_all_sessions(self):
        self.bci_recording = dict()
        self.bci_recording["data"] = np.concatenate([se["data"] for se in self.reshaped_sessions], axis=0)
        self.bci_recording["label"] = np.concatenate([se["mrk_class"] for se in self.reshaped_sessions], axis=0)
        self.bci_recording["timepoints"] = self.reshaped_sessions[0]["timepoints"]
        self.bci_recording["mnt"] = self.reshaped_sessions[0]["mnt"]

    def clear_all_sessions(self, type):
        if (type == 'filtered'):
            self.filtered_sessions.clear()
        elif (type == 'loaded'):
            self.loaded_sessions.clear()
        elif (type == 'time_filtered'):
            self.reshaped_sessions.clear()
        elif (type == 'all'):
            self.filtered_sessions.clear()
            self.loaded_sessions.clear()
            self.reshaped_sessions.clear()
        else:
            print('Not a valid argument, possible options are: filtered, loaded, all')


def load_bbci_data(data_path,
                   subjects_list,
                   sessions_list,
                   task_type,
                   time_interval,
                   merge_sessions,
                   reshape_type):
    data = []
    for subno, seno in zip(subjects_list, sessions_list):
        data.append(
            Subject(data_path=data_path, number=subno, sessionno=seno, task_type=task_type, time_interval=time_interval,
                    merge_sessions=merge_sessions, reshape_type=reshape_type))
    return data


def prepare_dataset(data,
                    norm_type,
                    norm_axis,
                    reshape_axes,
                    train_dev_test_split,
                    batch_size):
    X = np.concatenate([se.bci_recording["data"] for se in data], axis=0)
    Y = np.expand_dims(np.concatenate([se.bci_recording["label"] for se in data], axis=0), -1)
    # normalize all data
    x_norm, x_norm_params = normalize(X, norm_type=norm_type, axis=norm_axis)  # normalization over axis 1
    # input shape tuple  : (num_trial, num_electrodes, timepoints)
    if reshape_axes is not None:
        if len(reshape_axes) <= 2:
            raise Exception("input shape map should contain at least 2 indexes to perform permutation")
        else:
            x_norm = np.transpose(x_norm, reshape_axes)
    else:
        pass
    data = TensorDataset(torch.tensor(x_norm, dtype=torch.float32), torch.tensor(Y))
    train_loader, dev_loader, test_loader, num_samples = \
        create_data_loaders(data, train_dev_test_split=train_dev_test_split, batch_size=batch_size)
    return data, train_loader, dev_loader, test_loader, num_samples, x_norm_params
