# Environment artifacts

This folder contains files to help you reproduce the conda environment used to run this repo.

- environment.yml: exported conda environment (no build pins)
- requirements.txt: pip package list from the environment
- pytorch_cuda_info.txt: torch/CUDA availability snapshot (if torch is installed)
- gpu_list.txt and nvidia-smi.txt: GPU and driver/runtime info (if available)

## Recreate the environment

```bash
# Create env
conda env create -f env/environment.yml -n gnot_cuda11
conda activate gnot_cuda11

# (Optional) If pip packages are needed and not fully captured by conda export:
pip install -r env/requirements.txt

# Verify torch/CUDA (optional)
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda', torch.version.cuda)
print('is_cuda_available', torch.cuda.is_available())
print('device_count', torch.cuda.device_count())
PY
```

Notes:
- The CUDA version available depends on your driver and platform; if prebuilt wheels are unavailable, install matching CUDA toolkit or use CPU wheels.
- If you use a different env name, adjust commands accordingly.
