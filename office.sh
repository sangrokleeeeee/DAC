# #!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 python main.py data/office-home -d OfficeHome -s Ar Cl Rw -t Pr --log logs/dac/OfficeHome_Pr
CUDA_VISIBLE_DEVICES=2 python main.py data/office-home -d OfficeHome -s Ar Cl Pr -t Rw --log logs/dac/OfficeHome_Rw
CUDA_VISIBLE_DEVICES=2 python main.py data/office-home -d OfficeHome -s Ar Rw Pr -t Cl --log logs/dac/OfficeHome_Cl
CUDA_VISIBLE_DEVICES=2 python main.py data/office-home -d OfficeHome -s Cl Rw Pr -t Ar --log logs/dac/OfficeHome_Ar
