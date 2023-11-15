# NNESVM
Neural Networks as Control Variates in ESVM algorithm


## Running guide

To set up environment first run
```
git clone https://github.com/ArturGoldman/NNESVM
cd NNESVM
chmod u+x ./prep.sh
./prep.sh
```

To start training run
```
python ./train.py -c ./nn_esvm/configs/clean_configs/config_(setup name).json
```

It is also possible to generate chains in advance not to spend time on it during training.
To do so you have to run
```
python ./generate_samples.py -c ./nn_esvm/configs/clean_configs/config_genchains_(setup name).json
```
After this generated chains can be found in `./saved/data`. In order to use them, they should be specified
in config file. 

In order to understand the meanings of `.json` config files, see descriptions in:
- `./nn_esvm/configs/clean_configs/descriptive_config_launch.json` for experiment type run
- `./nn_esvm/configs/clean_configs/descriptive_config_genchains.json` for Markov Chain generation type run
___

In order to recreate experiments from paper, run

```
chmod u+x ./experiments.sh
./experiments.sh
```

## Specifics

- Testing procedure is implemented in parallel manner
- It was decided to use neural network to model function <img src="https://render.githubusercontent.com/render/math?math=\varphi">. 
Thus we had to calculate $\Delta\varphi$ in a differentiable manner.
Still, implementation allows one to experiment with various types of CV.
- Backprop through Empirical Spectral Variance was implemented in memory-efficient manner


## Results
1 layered Fully connected neural network with ReCU/Tanh activation was able to achieve acceptable variance reduction rate.

Details of training and experimenting can be found here: https://wandb.ai/artgoldman/NeuralESVM-HSE-HDI-V3


## Credits
- Structure of this repository is based on [pytorch-template](https://github.com/victoresque/pytorch-template) repository.
- Pima dataset located in `./saved/dataset.csv` was taken from https://www.kaggle.com/uciml/pima-indians-diabetes-database.
