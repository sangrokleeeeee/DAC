#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 python baseline.py data/domainnet -d DomainNet -s c i p r s -t q -i 7500 --lr 0.01 --log logs/dac/DomainNet_q
CUDA_VISIBLE_DEVICES=2 python baseline.py data/domainnet -d DomainNet -s c i p q s -t r -i 7500 --lr 0.01 --log logs/dac/DomainNet_r
CUDA_VISIBLE_DEVICES=2 python baseline.py data/domainnet -d DomainNet -s c i p q r -t s -i 7500 --lr 0.01 --log logs/dac/DomainNet_s