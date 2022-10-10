# bbcpy

#### Table of Contents

1. [Module and Folder Description](#module-description)
1. [Usage ](#usage)



## Module and Folder Description

The `models` module contains all the neural network architecture in separate python scripts. The neural network models should be implemented with the pytorch framework.

The `train` module represents the main module, where you will execute the corresponding script to train your model. The Training and evaluation pipeline is mainly with `Pytorch` framework implemented. Furthermore, we used `Ignite` library to facilitate code usage.
`Ignite` is a high-level pytorch wrapper that make training and evaluation more flexible and straightforward.  

The `Params` folder contains configuration files to automate model training and testing. The files should be in yaml format.

The `utils` module offers different utility scripts that are need for the project. 

    - `file_management` : tools and functions used for the configuration yaml file
    - `data_loader` : functions used to create, process and transform data

## Usage

To train a specific model, you need to follow these steps : 

### Step 1 : Setup Configuration
The Configuration of the pipeline is divided into multiple configs for each pipeline block:  
```
📦params
 ┣ 📂configs
 ┃ ┣ 📜data.yaml
 ┃ ┣ 📜logging.yaml
 ┃ ┣ 📜network.yaml
 ┃ ┣ 📜optims.yaml
 ┃ ┗ 📜tunning.yaml
 ┗ 📜baselines.yaml
```
- `baseline.yaml` is the master config file and contains all other configs using the `import` key.
- You can create multiple configuration setup using nested key declaration in the YAML syntax (use the `baselines.yaml` as template).
- You can derive a new configuration setup from default one and override  paramters by using the *&* and *<<* decorators (see the example)

  ```
  default_net: &default
    momentum:
      default: 0.1
      help:
    loss:
      default: "BCELoss"
      help: 
    activations:
      default: tanh
      help:
    hidden_size:
      default:
        - 32
        - 16
      help:
  default_net_relu:
    <<: *deep_net
    activations: relu
  ```
- Before run the script you need to specify :
  - baseline_yaml_name (e.g baselines)
  - baseline_mode (e.g debug_pipeline)
### Step 2 : Create a training pipeline 

Create a ``train.py`` in the **train** folder that has the same structure in the ``train_model_test_pipeline.py``, make sure then to adjust the parameters,  Hyperparameters and logging options in the **Ignite** and **Tensorboard** APIs.

#### To Test your training pipline : 

Run the script in the `train` folder with the corresponding configuration file as an argument.
```
python -m bbcpy.train.train_model_test_pipeline.py --file path_to_baseline.yaml
```

During the training, at each new experiement and at each new run a nested folder will be automatically created in the **runs** folder. 

Here is an example of the **runs** folder structure: 

```
📦runs
 ┗ 📂exp_debug
 ┃ ┗ 📂debug_run_00
 ┃ ┃ ┗ 📂2021-09-01_13-09-39
 ┃ ┃ ┃ ┣ 📂checkpoints
 ┃ ┃ ┃ ┃ ┣ 📜best_model_1_trial_1_validation_acc=0.3913.pt
 ┃ ┃ ┃ ┃ ┣ 📜best_model_1_trial_2_validation_acc=0.5652.pt
 ┃ ┃ ┃ ┃ ┣ 📜best_model_1_trial_3_validation_acc=0.2609.pt
 ┃ ┃ ┃ ┃ ┣ 📜best_model_1_trial_4_validation_acc=0.4783.pt
 ┃ ┃ ┃ ┃ ┗ 📜best_model_1_trial_5_validation_acc=0.4348.pt
 ┃ ┃ ┃ ┣ 📂tensorboard
 ┃ ┃ ┃ ┃ ┣ 📜events.out.tfevents.1630494588.user
 ┃ ┃ ┃ ┣ 📜Hparams.yaml
 ┃ ┃ ┃ ┣ 📜model_summary.txt
 ┃ ┃ ┃ ┗ 📜test_study.db
```

### Step 3: Monitor your Model 

To track you training and evaluation with **Tensorbaord**, run the command in terminal: 

```
  tensorboard --logdir runs/exp_name
```

### Step 4 : Running on the IDA Cluster (WIP)

#### Submitting jobs 
Create a `job.sh` in the folder **jobs**.

```
qsub -N "my funny script for the tesla gpu" /PATH_TO_FOLDER_PROJECT/bbcpy/jobs/first_job.sh
```

#### Monitoring jobs

```
qstat [options]
```
* -u user
Typical usage:
-u $USER
Print all jobs of a given user, print all my jobs..
* -j job-id
Prints full information of the job with the given job-id.
* -f
Prints all queues and jobs.
* -help
Prints all possible qstat options.

#### Deleting a job

```
qdel job-id
```

### Step 5 : Hyperparameters Tuning 

We are using here a framework for Auto-Tuning Hyperprameters **Optuna**

To Select Hyperparmaters for tuning, we need to add `trial` and select the `optimizer` output metric. Additionally, you need to creat a `pruner` for a trial determination strategy.Finnaly, you need to setup the `study` function.   

You can find a test example in the ``nn.models.py``  and `train_pipeline_supervised.py`. Make sure to adjust the parameters in the `baseline.yml` configuration file 

You need to adjust your model with the desired model hyperparmeters. 

For detailed informations, check the [Optuna](https://optuna.readthedocs.io/en/stable/) documentation and watch the short tutorial [Auto-Tuning Hyperparameters with Optuna and PyTorch](https://www.youtube.com/watch?v=P6NwZVl8ttc&t=4s&ab_channel=PyTorch)

To open the optuna Dashboard: 
```
optuna-dashboard sqlite:///PATH_TO_FILE/test_study.db
 ```