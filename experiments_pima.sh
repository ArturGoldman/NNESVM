#!/bin/bash

#python3 ./train.py -c ./nn_esvm/configs/config_pima4.json -r ./saved/models/pima/0603_121111/model_best.pth
python3 ./train.py -c ./nn_esvm/configs/config_pima3.json
python3 ./train.py -c ./nn_esvm/configs/config_pima4.json
