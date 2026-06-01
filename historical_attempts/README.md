# Historical Attempts

This folder stores scripts that were tried during development but are not the current highest-scoring route.

The recorded best pipeline is kept in the repository root:

```text
dataset.py
submission_utils.py
train_dinov2_lora.py
train_dinov2_lora_v2.py
```

`train_dinov2_lora_v2.py` is the current best route.

Historical scripts include:

```text
SAM / CLIP preprocessing
frozen DINOv2 feature classifiers
feature ensembles
INR / SIREN descriptors
Meta-INR
K-fold multi-seed INR rank averaging
label propagation
kNN / nearest-neighbor analysis
weighted voting
```

They are kept for reproducibility and presentation discussion. The current best public score comes from DINOv2 LoRA v2 / PEFT fine-tuning on original images. INR is included here as a tried-but-unused method, not as part of the final pipeline.

Some historical scripts import helper modules that now live in this folder. If a direct script run cannot find imports, run from the repository root with:

```powershell
$env:PYTHONPATH=".;historical_attempts"
```
