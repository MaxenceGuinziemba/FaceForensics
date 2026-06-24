# Deepfake Detection with EfficientNet on FaceForensics++

![Pipeline](images/teaser.png)

Binary deepfake detector trained on the [FaceForensics++](https://arxiv.org/abs/1901.08971) dataset. The model classifies face crops as **real** or **fake** across four manipulation methods (Deepfakes, Face2Face, FaceSwap, NeuralTextures).

**Best result:** EfficientNet-B0 achieves **0.9671 AUC** on FF++ c23, exceeding the [DeepfakeBench](https://arxiv.org/abs/2307.01426) reference of 0.9567 for EfficientNet-B4 (a 3.6x larger model).

## Results

| Version | Model | Val Acc | Test Acc | AUC |
|---------|-------|---------|----------|-----|
| V1 | ResNet18 (11M) | 65.3% | - | - |
| V4 | EfficientNet-B0 (5.3M) | 89.7% | **89.8%** | **0.967** |
| V6 | EfficientNet-B4 (19M) | **92.1%** | 89.4% | 0.959 |

## Project Structure

```
src/                    Source code
  models/               EfficientNet/ResNet architectures
  data/                 Dataset, transforms, augmentations, face extraction, mixup
  train.py              Training script
  evaluate.py           Evaluation (accuracy, AUC, ROC, confusion matrix)
  detect.py             Inference on video
scripts/                SLURM job scripts and utilities
configs/splits/         Train/val/test splits (360/70/70 video pairs)
results/                Training curves, ROC curves, confusion matrices per version
images/                 Paper figures (used in the report)
report.typ              Project report (Typst)
```

## Quick Start

```bash
pip install -r requirements.txt

# Train (EfficientNet-B0, local test)
python -m src.train --data_root data --compression c40 --model resnet18 \
    --epochs 2 --batch_size 4 --num_workers 0

# Train (GPU cluster, full dataset)
sbatch scripts/submit_train_v4.sh    # EfficientNet-B0
sbatch scripts/submit_train_v6.sh    # EfficientNet-B4

# Evaluate
python -m src.evaluate --checkpoint checkpoints/best_model.pth \
    --data_root data --compression c23
```

## Dataset

The FaceForensics++ dataset is not included. Request access from the [original authors](https://github.com/ondyari/FaceForensics), then download:

```bash
python scripts/download_dataset.py ./data -d all -c c23 -t videos --server EU2
```

## References

- Rossler et al. "FaceForensics++: Learning to Detect Manipulated Facial Images." ICCV 2019.
- Yan et al. "DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection." NeurIPS 2023.
- Tan, Le. "EfficientNet: Rethinking Model Scaling for CNNs." ICML 2019.
