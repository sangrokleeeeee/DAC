# #!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4 python main.py data/terra -d Terra -s L43 L46 L100 -t L38 --log logs/dac/terra_38 
CUDA_VISIBLE_DEVICES=4 python main.py data/terra -d Terra -s L38 L46 L100 -t L43 --log logs/dac/terra_43 
CUDA_VISIBLE_DEVICES=4 python main.py data/terra -d Terra -s L38 L43 L100 -t L46 --log logs/dac/terra_46 
CUDA_VISIBLE_DEVICES=4 python main.py data/terra -d Terra -s L38 L43 L46 -t L100 --log logs/dac/terra_100