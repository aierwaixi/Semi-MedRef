# Semi-MedRef

Official implementation of *Semi-MedRef: Semi-Supervised Medical
Referring Image Segmentation with Cross-Modal Alignment*.

The code supports MMI-UNet and GuideDecoder backbones. The default method uses
an EMA teacher--student framework with PosAug, T-PatchMix, and positional
affinity contrastive learning (PACL). ISPG is provided as an optional
robustness extension and is disabled in the main configurations.

## Installation

```bash
conda env create -f environment.yml
conda activate semimedref
bash scripts/download_pretrained.sh
```

The first run may download CXR-BERT and ConvNeXt-Tiny from Hugging Face.

## Data

Download QaTa-COV19 and MosMedData+ from their official sources. The
language-guided annotations are available from the
[LViT repository](https://github.com/HUANGLIZI/LViT). PosMed preprocessing uses
the public [PRS-Med dataset](https://huggingface.co/datasets/huyquoctrinh/PRS-Med).

Place QaTa-COV19 and MosMedData+ under:

```text
datasets/
├── QaTa/
│   ├── train/{images,masks}/
│   ├── valid/{images,masks}/
│   └── test/{images,masks}/
└── MosMed/
    ├── frames/
    └── masks/
```

The fixed data manifests used by the experiments are included in
`data_splits/`.

## Training

Run all commands from the package root.

```bash
# QaTa-COV19
python train_semi.py --config configs/qata_mmiunet.yaml --ratio 0.02 --device 0
python train_semi.py --config configs/qata_guidedecoder.yaml --ratio 0.02 --device 0

# MosMedData+
python train_semi.py --config configs/mosmed_mmiunet.yaml --ratio 0.05 --device 0
python train_semi.py --config configs/mosmed_guidedecoder.yaml --ratio 0.05 --device 0
```

Use `--seed`, `--output-dir`, `--num-workers`, or `--resume` when needed.
Training settings are defined in the corresponding YAML configuration files.

## PosMed Preprocessing

Arrange PosMed as follows:

```text
datasets/PosMed/
├── annotations/qa/
└── data/
```

Generate the deterministic 5% and nested 15% partitions:

```bash
python scripts/prepare_posmed_splits.py \
  --qa-dir datasets/PosMed/annotations/qa \
  --data-root datasets/PosMed/data \
  --brain-pid-map data_splits/PosMed/brain_mri_slice_to_pid.csv \
  --output-dir /tmp/posmed_r0.05 \
  --labeled-fraction 0.05 \
  --seed 42

python scripts/prepare_posmed_splits.py \
  --qa-dir datasets/PosMed/annotations/qa \
  --data-root datasets/PosMed/data \
  --brain-pid-map data_splits/PosMed/brain_mri_slice_to_pid.csv \
  --output-dir /tmp/posmed_r0.15 \
  --labeled-fraction 0.15 \
  --required-labeled-manifest /tmp/posmed_r0.05/train_labeled_5pct.json \
  --seed 42
```

The script uses the free-form answer as the model text, preserves recoverable
patient or case groups, and checks for exact image overlap between partitions.
The fixed manifests are also included in `data_splits/PosMed/`.

## Optional ISPG

ISPG is used only for robustness experiments with missing positional
expressions. Its configuration and entry points are:

```text
configs/*_ispg.yaml
train_position_predictor.py
train_semi_soft_position.py
```

## License

This code is released under the GNU GPLv3 license. Datasets, pretrained models,
and upstream components retain their original licenses.
