#!/bin/bash

# Funnel distribution experiments
python ./generate_samples.py -c ./nn_esvm/configs/clean_configs/config_genchains_funnel_2.json
python ./train.py -c ./nn_esvm/configs/clean_configs/config_funnel_2_recu.json
python ./train.py -c ./nn_esvm/configs/clean_configs/config_funnel_2_lin.json
python ./train.py -c ./nn_esvm/configs/clean_configs/config_funnel_2_relu.json
python ./train.py -c ./nn_esvm/configs/clean_configs/config_funnel_2_requ.json

# Banana distribution experiment
python ./generate_samples.py -c ./nn_esvm/configs/clean_configs/config_genchains_banana_6.json
python ./train.py -c ./nn_esvm/configs/clean_configs/config_banana_6_recu.json


# Pima dataset logistic regression experiment
python ./generate_samples.py -c ./nn_esvm/configs/clean_configs/config_genchains_pima.json
python ./train.py -c ./nn_esvm/configs/clean_configs/config_pima_tanh.json
