#!/bin/bash

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1wQMYS4Cot4GPEQ-n3MU8cJBCH1BrynE9' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1wQMYS4Cot4GPEQ-n3MU8cJBCH1BrynE9" -O ./NV/my_model.pth && rm -rf /tmp/cookies.txt

python3 ./NV/test.py -c ./NV/hw_nv/config_test.json -r ./NV/my_model.pth