# CSL Sign Language Classifier — Skeleton-based Training

This repository contains a training script for classifying sign/gesture sequences from **skeleton files** (keypoint sequences).  
It supports **transfer learning** by loading a pretrained checkpoint from another sign language dataset and uses clean, configurable paths.

---

## Features
- Train/validate loop with Top-1 / Top-5 accuracy
- **Skeleton-based inputs** via your `dataset.loadedDataset`
- Path-only configuration (no hard-coded Windows paths)
- **Pretrained weights** via `--pretrained` (from another SL dataset)
- Auto-resume if `checkpoint.pth.tar` exists in the run folder
- TensorBoard logging
- Best + last checkpoints per run

---

## Requirements
- Python 3.8+
- `torch`, `torchvision`
- `numpy`, `scikit-learn`
- `tensorboard`

Install:
```bash
pip install torch torchvision numpy scikit-learn tensorboard
```

> The script expects your existing modules:
> - `model.MLP`
> - `dataset.loadedDataset`  ← **loads skeleton files**
> - `utils.AverageMeter`

---

## Data (Skeleton Files)

`loadedDataset` should read **skeleton sequences** from disk given `--train-dir` and `--val-dir`.  
A common layout is:

```
/data/CSL8_KP70/
├─ train/
│  ├─ CLASS_001/
│  │  ├─ sample_0001.npy
│  │  └─ sample_0002.npy
│  └─ CLASS_002/
│     ├─ sample_0003.npy
│     └─ ...
└─ test/
   ├─ CLASS_001/
   │  └─ sample_0999.npy
   └─ CLASS_002/
      └─ ...
```

**Typical file contents** (exact details depend on your `loadedDataset`):
- `numpy` array with shape like:
  - `(T, K, 2)` → 2D keypoints `(x, y)`
  - `(T, K, 3)` → 2D + confidence `(x, y, c)`
  - `(T, K, 3D)` for 3D if applicable  
- Where:
  - `T` = number of frames
  - `K` = number of keypoints/joints (e.g., 70)
- Your `loadedDataset` returns a batch as a **list of tensors** (e.g., joints/bones/vel) and an integer label. The model uses the **last timestep** logits (`output[:, -1, :]`) for classification—this is preserved.

If your skeletons are stored as `.pt`, `.json`, or in multi-stream folders, keep that — the script just delegates to `loadedDataset`.

---

## Usage

### 1) Fresh run (new checkpoint directory)
```bash
python trainCSL.py   --train-dir /data/CSL8_KP70/train   --val-dir   /data/CSL8_KP70/test    --model     ./save_model   --run-name  chinese8KP   --epochs    200   --batch-size 32   --lr 1e-5   --lr-step 200
```

### 2) Start from a **pretrained** checkpoint (from another sign language)
```bash
python trainCSL.py   --train-dir /data/CSL8_KP70/train   --val-dir   /data/CSL8_KP70/test    --model     ./save_model   --run-name  chinese8KP_from_pretrain   --pretrained /checkpoints/phoenix2014T_best.pth.tar   --epochs 200 --batch-size 32 --lr 1e-5 --lr-step 200
```
- Loads weights with `strict=False` so classifier-size mismatches are OK.
- New checkpoints go to `./save_model/chinese8KP_from_pretrain/`.

### 3) Resume training an existing run (auto-detect)
If `./save_model/chinese8KP/checkpoint.pth.tar` exists, just re-run:
```bash
python trainCSL.py   --train-dir /data/CSL8_KP70/train   --val-dir   /data/CSL8_KP70/test    --model     ./save_model   --run-name  chinese8KP
```

### 4) Windows (PowerShell) example
```powershell
python .	rainCSL.py `
  --train-dir "D:\Keren_SLR\CSL8_KP70	rain" `
  --val-dir   "D:\Keren_SLR\CSL8_KP70	est"  `
  --model     ".\save_model" `
  --run-name  "chinese8KP" `
  --epochs 200 --batch-size 32 --lr 1e-5 --lr-step 200
```

### 5) TensorBoard
```bash
tensorboard --logdir ./save_model/chinese8KP/tb
```

---

## Outputs
For a run named `<run-name>`, files are written to `./save_model/<run-name>/`:
- `checkpoint.pth.tar` — last checkpoint (updated every epoch)
- `<run-name>_best.pth.tar` — best checkpoint by validation Top-1
- `tb/` — TensorBoard logs
- `results.txt` — appends best result summaries by epoch

---

## Command-line Arguments

| Argument        | Type   | Default           | Description                                        |
|-----------------|--------|-------------------|----------------------------------------------------|
| `--train-dir`   | str    | `./data/train`    | Training dataset root (skeleton files)             |
| `--val-dir`     | str    | `./data/val`      | Validation/Test dataset root (skeleton files)      |
| `--model`       | str    | `./save_model`    | Root folder to save runs                           |
| `--run-name`    | str    | `chinese8KP`      | Subfolder under `--model` for this run             |
| `--pretrained`  | str    | `''`              | Path to pretrained checkpoint `.pth.tar`           |
| `--arch`        | str    | `MLP`             | Model architecture label (for logging only)        |
| `--rnn-layers`  | int    | `1`               | Number of RNN layers                               |
| `--hidden-size` | int    | `3000`            | RNN hidden size                                    |
| `--fc-size`     | int    | `2000`            | Fully-connected layer size before RNN              |
| `--epochs`      | int    | `2000`            | Number of epochs                                   |
| `--lr`          | float  | `1e-5`            | Initial learning rate                              |
| `--lr-step`     | int    | `2000`            | LR decay step (epochs); multiplies LR by 0.1       |
| `--batch-size`  | int    | `32`              | Mini-batch size                                    |
| `--workers`     | int    | `0`               | DataLoader workers                                 |

> **Note:** The script will resume automatically if `checkpoint.pth.tar` is present in the run folder.

---

## Tips
- Use different `--run-name` values to keep experiments separate.
- For cross-dataset transfer (pretraining on another sign language), pass `--pretrained` and optionally freeze early layers in your model (minor code tweak if desired).

---
---

## Citation
If you use this repository, please cite the following paper:

```bibtex
@InProceedings{10.1007/978-3-031-70239-6_16,
author="Artiaga, Keren
and Lynch, Conor
and Afli, Haithem
and Hasanuzzaman, Mohammed",
editor="Rapp, Amon
and Di Caro, Luigi
and Meziane, Farid
and Sugumaran, Vijayan",
title="The Influence of Iconicity in Transfer Learning for Sign Language Recognition",
booktitle="Natural Language Processing and Information Systems",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="226--240",
abstract="Most sign language recognition research relies on Transfer Learning (TL) from vision-based datasets such as ImageNet. Some extend this to alternatively available language datasets, often focusing on signs with cross-linguistic similarities. This body of work examines the necessity of these likenesses on effective knowledge transfer by comparing TL performance between iconic signs of two different sign language pairs: Chinese to Arabic and Greek to Flemish. Google Mediapipe was utilised as an input feature extractor, enabling spatial information of these signs to be processed with a Multilayer Perceptron architecture and the temporal information with a Gated Recurrent Unit. Experimental results showed a 7.02{\%} improvement for Arabic and 1.07{\%} for Flemish when conducting iconic TL from Chinese and Greek respectively.",
isbn="978-3-031-70239-6"
}
```


## License
Add your license here (e.g., MIT).
