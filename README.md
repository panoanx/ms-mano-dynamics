# MS-MANO Muscle Inverse Dynamics 

This repository contains the code for training the muscle inverse dynamics of the MS-MANO hand model as described in the CVPR'24 paper [MS-MANO: Enabling Hand Pose Tracking with Biomechanical Constraints](https://ms-mano.robotflow.ai/). Please read section 4.1 and 5.4 of the paper for more details.

For other components of the MS-MANO project, please refer to the [project page of MS-MANO](https://ms-mano.robotflow.ai/).

## Installation

Clone the repository using the following command:
```sh
# through ssh
git clone git@github.com:panoanx/ms-mano-dynamics.git
# or through https
# git clone https://github.com/panoanx/ms-mano-dynamics.git 
```

We recommend using [pyenv](https://github.com/pyenv/pyenv) for managing Python versions and [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) for creating virtual environments.
The code is tested with Python 3.10.12.

Install the dependencies:
```sh
pip install -r requirements.txt
```

> [!NOTE]
> The `stable_baselines3` package requires `wandb` to work properly. 
> You can disable it by
> ```sh
> wandb disabled
> ```

## Data for Mimicking
We provide an example data tracjectories for training the network. The data is recorded using a data glove. You can download the file from [here](https://r2.robotflow.ai/ms_mano_data_glove_pd.tar.gz).

The `mim_data_path` in the configuration file should be set to the path of the extracted folder.

## Train the IDNet

Please make sure you have the [MS-MANO Simulator](https://github.com/panoanx/ms-mano-unity) installed.

> [!TIP]
> You may want first understand how the simulator communicates with the training code. 
> 
> In short, at each simulation step, the Unity simulator sends the user-specified features to python via a TCP socket. After the python code (defined as a gym step) processes the features, it sends the actions back to the simulator. 
> If you want to implement a new function in the simulator and returns new data structures, both the unity-side and python-side code should be updated.
> 
> Please refer to the [README](https://docs.robotflow.ai/pyrfuniverse/markdown/introduction.html).

To train the IDNet, navigate to the root directory of this repository and run the following command:
```sh
python train/idnet.py
```

We use [confz](https://github.com/Zuehlke/ConfZ) for managing the configuration files. The default configuration file is `config/idnet_1024_512.yaml`. You can modify the values for testing.

After you have started the python script, you have to start the simulators. The script will automatically connect to the simulators.

### If you are using the builds
The python script will start the simulators if it's correctly configured in the configuration file. 
```yaml
executable_file: "path/to/RFUniverse.x86_64"
```

### If you are using the Unity Editor
You have to start the Unity Editor manually by clicking the play button. 
If the editor does not automatically connect to the python script, you possibly shall ensure the unity starts after the python script pops out 'Waiting for connection...'.


> [!IMPORTANT]
> To ensure efficient training, we recommend you to use the builds of the simulator (instead of running in Unity Editor) for parallelization.
> 
> As Unity is single-threaded (not optimized for multi-core), we manually distribute the agents to multiple processes. 
> You can turn on this feature by setting `use_subproc: true` in the configuration file. 
> The number of process `n_proc` represents the number of unity instances, and the number of agents `num_agents` represents the number of agents in each unity instance.
> The total number of agents is `n_proc * num_agents`.
>
> You may notice we use both `n` and `num` in the configuration file.
> The `n_proc` naming follows the linux convention, while the `num_agents` follow the gym convention. 

> [!CAUTION]
> We have tested the code in several platforms. However, there is still vulnerability. We have been reported in some cases the simulator sub-processes may not be terminated properly after the main python script is terminated. It's recommended to check the processes after the training is finished using process management tools and manually kill the processes if necessary.

## Results, Logs, and Checkpoints

You can find several checkpoints in the `models` directory and we attached an example training output in the `logs` directory. 

To load the trained model, you can refer to the official [stable-baselines3 documentation](https://stable-baselines3.readthedocs.io/en/master/guide/save_format.html). 

You can also visualize the loss and reward curves using `tensorboard`. The tensorboard files are located in the `tensorboard` directory. 

You may notice we a high rollout-reward-mean value in the first step. This is due to the initializations of the muscles. We recommend you to ignore the first step when analyzing the results.

