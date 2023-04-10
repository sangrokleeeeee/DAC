#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3 python main.py data/domainnet -d DomainNet -s i p q r s -t c -i 7500 --lr 0.01 --log logs/dac/DomainNet_c
CUDA_VISIBLE_DEVICES=3 python main.py data/domainnet -d DomainNet -s c p q r s -t i -i 7500 --lr 0.01 --log logs/dac/DomainNet_i
CUDA_VISIBLE_DEVICES=3 python main.py data/domainnet -d DomainNet -s c i q r s -t p -i 7500 --lr 0.01 --log logs/dac/DomainNet_p