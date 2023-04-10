# #!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3 python main.py data/VLCS -d VLCS -s V L C -t S --log logs/dac/VLCS_S
CUDA_VISIBLE_DEVICES=3 python main.py data/VLCS -d VLCS -s V L S -t C --log logs/dac/VLCS_C
CUDA_VISIBLE_DEVICES=3 python main.py data/VLCS -d VLCS -s V C S -t L --log logs/dac/VLCS_L
CUDA_VISIBLE_DEVICES=3 python main.py data/VLCS -d VLCS -s L C S -t V --log logs/dac/VLCS_V