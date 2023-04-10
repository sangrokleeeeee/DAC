# #!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python main.py data/PACS -d PACS -s A C S -t P --log logs/dac/PACS_P
CUDA_VISIBLE_DEVICES=1 python main.py data/PACS -d PACS -s P C S -t A --log logs/dac/PACS_A
CUDA_VISIBLE_DEVICES=1 python main.py data/PACS -d PACS -s P A S -t C --log logs/dac/PACS_C
CUDA_VISIBLE_DEVICES=1 python main.py data/PACS -d PACS -s P A C -t S --log logs/dac/PACS_S