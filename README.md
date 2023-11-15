# NNESVM
Neural Netowrks as Control Variates in ESVM algorithm
This is a branch with an old version of the algorithm and code.

## Running guide

To set up environment fisrt run
```
git clone https://github.com/ArturGoldman/NNESVM
chmod u+x ./NNESVM/prep.sh
./NNESVM/prep.sh
```

To start training run
```
python ./NNESVM/train.py -c ./NNESVM/nn_esvm/configs/config_(setup name).json
```

It is also possible to generate chains in advance not to spend time on it during training.
To do so you have to run
```
python ./NNESVM/generate_samples.py -c ./NNESVM/nn_esvm/configs/config_genchains.json
```
After this generated chains can be found in `./NNESVM/saved/data`. In order to use them, they should be specified
in config file. Compare for example `config_gmm.json` and `config_funnel_2.json`.

## Specifics

- Testing procedure is implemented in parallel manner
- It was decided to approximate <img src="https://render.githubusercontent.com/render/math?math=\nabla\varphi"> instead of plain <img src="https://render.githubusercontent.com/render/math?math=\varphi"> in order to avoid one more backprop through model gradient in differentiable manner
- Backprop through Spectral Variance was implemented in memory-efficient manner


## Results
3 layered Fully connected neural network was able to achieve acceptable variance reduction rate.

Details of training and experimenting can be found here: https://wandb.ai/artgoldman/NeuralESVM-HSE-HDI


## Credits
Structure of this repository is based on [pytorch-template](https://github.com/victoresque/pytorch-template) repository.
