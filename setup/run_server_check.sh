#!/bin/bash
# Wait for the live server, then run the full server-side check suite.
set -e
bash /mnt/c/Users/sng51/Desktop/WebVLA/setup/wait_live.sh
cd /mnt/c/Users/sng51/Desktop/WebVLA
bash setup/run.sh live/_server_check.py 2>&1 | tr -d '\0' | grep -vE \
"wsl:|WARNING|warn|deprecat|Deprecat|Gym|gymnasium|migration|TensorRT|oneDNN|AVX2|NUMA|cuDNN|cuFFT|cuBLAS|tokeniz|padding|truncat|resume_download|_pytree|SentryHub|packrat|parseString|resetCache|enablePackrat|ParserElement|mplstyle|task orders|torch.load|init_states|FutureWarning|weights_only|pickle|allowlisted|upgrade|See the|feature|recommend|software|additional|subprocess|Created device"
