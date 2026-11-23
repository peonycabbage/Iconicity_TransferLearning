# Iconicity in Transfer Learning for Sign Language Recognition

Research code accompanying *The Influence of Iconicity in Transfer Learning for Sign Language Recognition* (NLDB 2024). The study compared transfer between Chinese and Arabic Sign Language and between Greek and Flemish Sign Language. The scripts here cover CSL preprocessing and a configurable keypoint-based sign classifier. The paper describes the full experimental setup.

## Setup

Install PyTorch and torchvision for your platform, then install the remaining dependencies. The keypoint extractor requires a MediaPipe build that exposes the legacy Holistic API (`mediapipe.solutions.holistic`).

```bash
python -m pip install numpy tensorboard opencv-python "mediapipe==0.10.21"
```

## Prepare the keypoints

Organise videos under `data/videos/<class>/<video-file>`. Run the commands from the repository root.

```bash
python CSL_convert2frames.py --input-dir data/videos --output-dir data/frames
python convert2keypCSL.py --input-dir data/frames --output-dir data/keypoints
```

The extractor saves one NumPy file per frame under `data/keypoints/<class>/<clip>/`. Each file contains 138 values from both hands, the shoulders and the wrists. Place the clip folders in separate training and validation roots, such as `data/train/<class>/<clip>/` and `data/val/<class>/<clip>/`. Both splits should have the same class folders because the loader assigns class IDs independently using sorted folder names.

## Train and transfer

```bash
python train.py --train-dir data/train --val-dir data/val --run-name csl-baseline --epochs 200
python train.py --train-dir data/train --val-dir data/val --run-name csl-transfer --epochs 200 --pretrained checkpoints/source.pth.tar
```

The transfer run loads parameter tensors whose names and shapes match the target model. If the number of classes changes, the classifier head starts with newly initialised weights. Source checkpoints must contain a `state_dict`. Use separate run names for separate experiments. Reusing a run name loads the model and epoch from its latest checkpoint when one exists.

Each run writes its latest checkpoint, its best checkpoint by validation Top-1 accuracy when accuracy improves, a results log and TensorBoard events to `save_model/<run-name>/`. Run `python train.py --help` for training options.

```bash
tensorboard --logdir save_model
```

## Citation

```bibtex
@inproceedings{artiaga2024iconicity,
  author    = {Artiaga, Keren and Lynch, Conor and Afli, Haithem and Hasanuzzaman, Mohammed},
  title     = {The Influence of Iconicity in Transfer Learning for Sign Language Recognition},
  booktitle = {Natural Language Processing and Information Systems},
  year      = {2024},
  pages     = {226--240},
  publisher = {Springer Nature Switzerland},
  doi       = {10.1007/978-3-031-70239-6_16}
}
```

## License

Released under the [MIT License](LICENSE).
