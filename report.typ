#set page(
  header: [ _Maxence Guinziemba_ #h(1fr) Integrative AI Project –- 2A ],
  footer: [
    #context [ Page #counter(page).display() #h(1fr) ]
  ]
)
#set heading(numbering: "1.1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#show link: set text(fill: blue)

#linebreak() #linebreak() #linebreak()
#align(center)[
  #set text(22pt, weight: "bold")
  Deepfake Detection with EfficientNet \
  on FaceForensics++
]
#v(6pt)
#align(center)[
  #set text(14pt)
  Maxence Guinziemba
]
#align(center)[
  #set text(12pt)
  Telecom Paris – 2025-2026 \
  Integrative AI Project – 2A
]
#v(4pt)
#align(center)[
  #set text(11pt, style: "italic")
  Based on: Rossler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images" (ICCV 2019)
]
#v(8pt)
#outline(depth: 2, indent: 1em)
#pagebreak()

= Introduction

The rapid advancement of deep learning-based face manipulation techniques poses a significant threat to the integrity of visual media. Methods such as Deepfakes, Face2Face, FaceSwap, and NeuralTextures can produce highly realistic forged videos that are increasingly difficult to distinguish from authentic content, raising concerns about misinformation, identity fraud, and erosion of trust in digital media.

The objective of this project is to implement a binary deepfake detector (real vs. fake) by reproducing and extending the methodology presented in the FaceForensics++ benchmark. Starting from the original codebase (which provided the model architecture and dataset splits but lacked training and evaluation scripts), a complete training pipeline was built from scratch. Through seven iterative training runs on a GPU cluster (RTX 3090, Telecom Paris), the pipeline was progressively refined, ultimately achieving an AUC of *0.9671* with EfficientNet-B0 (5.3M parameters) and *0.9587* with EfficientNet-B4 (19M parameters), both exceeding the DeepfakeBench reference of 0.9567 for EfficientNet-B4 on FF++ c23.

= Related Work

Detecting deepfakes is fundamentally an image classification problem: given a face, the model must decide whether it is authentic or manipulated. The challenge lies in the subtlety of modern manipulations: artifacts are often invisible to the human eye, especially after video compression. Two key references frame this project.

== FaceForensics++ (Rossler et al., 2019)

FaceForensics++ established the standard benchmark for this task. The authors collected 1,000 original YouTube videos and applied four manipulation methods to each, creating a diverse test bed:

- *Deepfakes*: an autoencoder swaps one person's face onto another's body.
- *Face2Face*: transfers facial expressions from a source to a target video in real time.
- *FaceSwap*: a graphics-based approach that replaces faces using 3D model fitting.
- *NeuralTextures*: modifies facial textures using a neural network, producing the most subtle artifacts.

Each video is provided at three compression levels (raw, high quality c23, low quality c40), since real-world videos are almost always compressed.

#figure(
  image("images/teaser.png", width: 75%),
  caption: [FaceForensics++ pipeline (from the original paper). Left: collection. Center: manipulation via reenactment or replacement. Right: binary CNN detection.],
)

The authors trained XceptionNet (23M parameters) and achieved *95.91% accuracy* on c23, using class weighting to handle the 1:4 real-to-fake imbalance and a freeze/unfreeze transfer learning strategy.

== DeepfakeBench (Yan et al., NeurIPS 2023)

DeepfakeBench later unified the evaluation of multiple detectors on this dataset. Their results on FF++ c23 provide the reference points we aim to match or exceed:

#figure(
  table(
    columns: 2,
    align: (left, center),
    [*Detector*], [*AUC*],
    [UCF (XceptionNet variant)], [0.9705],
    [XceptionNet], [0.9637],
    [EfficientNet-B4], [0.9567],
  ),
  caption: [DeepfakeBench results on FF++ c23. These are our targets to beat.],
)

= Methodology

== Problem Setup

The task is binary classification: given a video, determine whether the face it contains is *real* or *fake*. Since neural networks operate on images rather than videos, the problem reduces to classifying individual face crops extracted from video frames. The overall pipeline follows four steps:

#align(center)[
  #set text(10pt)
  #box(stroke: 0.5pt, inset: 8pt, radius: 4pt)[
    Video #sym.arrow.r Face detection #sym.arrow.r Crop & augment #sym.arrow.r CNN classifier #sym.arrow.r Real / Fake
  ]
]

== Transfer Learning: Why Not Train from Scratch?

Training a deep CNN from scratch requires millions of images. The FaceForensics++ dataset contains only 2,160 training videos (far too few). Instead, we use *transfer learning*: take a model pre-trained on ImageNet (1.2 million natural images, 1000 categories) and adapt it to our binary task. The model already knows how to detect edges, textures, and shapes. We only need to teach it which patterns indicate manipulation.

This is done in two phases:

+ *Freeze phase* (first $N$ epochs): the pre-trained backbone is frozen, and only a small classifier head is trained on top. This gives the classifier time to learn how to use the existing features without corrupting them.

+ *Unfreeze phase*: the entire network becomes trainable. The backbone can now fine-tune its features to detect deepfake-specific artifacts (compression boundaries, blending seams, texture inconsistencies).

== Data Augmentation and Regularization

With a limited dataset, there is a high risk of *overfitting*. The model memorizing training examples rather than learning general patterns. Several techniques are used to prevent this:

- *Data augmentation*: training images are randomly flipped, rotated, color-shifted, and compressed to simulate real-world variation. The model sees a slightly different version of each image every epoch, forcing it to learn robust features rather than memorizing pixel patterns.

- *Mixup*: two training images are blended together (e.g., 70% image A + 30% image B), and the label is blended accordingly. This encourages the model to learn smoother decision boundaries.

- *Label smoothing*: instead of training the model with hard labels (0 or 1), we use soft labels (0.05 or 0.95). This prevents the model from becoming overconfident in its predictions.

- *Class weighting*: the dataset contains 4 times more fake videos than real ones. Without weighting, the model learns a shortcut: predict "fake" most of the time and achieve 80% accuracy without learning any real features. Inverse-frequency weighting penalizes mistakes on the minority class proportionally.

- *BatchNorm freeze*: normalization layers in the pre-trained model carry statistics learned from millions of ImageNet images. When fine-tuning with small batches (16 to 32 images), the batch-level statistics are noisy and can overwrite these carefully learned values. Keeping BatchNorm in evaluation mode preserves the pre-trained statistics.

== Face Extraction

Raw video frames contain the full scene (background, body, etc.), but the manipulation only affects the face. Feeding full frames to the classifier forces it to learn to ignore irrelevant pixels, wasting capacity. An OpenCV-based face detector extracts and crops the face region from each frame, with a small margin around it (1.3$times$ the bounding box).

In our early versions, face detection ran on-the-fly during training, adding ~1000 seconds per epoch. Later versions pre-extract 30--50 face crops per video to disk, reducing epoch time from ~1400s to ~350s.
= Experiments and Iterations

The pipeline was refined through six major iterations on a single RTX 3090 (24GB VRAM) at Telecom Paris, trained on the FF++ c23 dataset (2,160 training and 420 validation videos). Each version built on the previous one, addressing a specific failure or limitation uncovered during training. Version 2 is omitted as its results were equivalent to Version 1.

#block(
  width: 100%,
  table(
    columns: 5,
    align: (left, left, center, center, left),
    [*Version*], [*Model*], [*LR*], [*Best Val*], [*Key additions*],
    [V1], [ResNet18 (11M)], [2e-04], [65.3%], [Baseline, 5 images, no augmentation, no freeze],
    [V3], [EffNet-B0 (5.3M)], [3e-05], [73.8%], [freeze/unfreeze, mixup, face extraction, warmup],
    [V4], [EffNet-B0 (5.3M)], [3e-05], [*89.7%*], [bug fix, ReduceLROnPlateau, assertion],
    [V5], [EffNet-B4 (19M)], [1e-05], [90.9%], [pre-extracted faces (30/video), batch 16],
    [V6], [EffNet-B4 (19M)], [3e-05], [*92.1%*], [380$times$380, class weights, BN freeze, uniform LR],
  ),
)

== Version 1 : The Overfitting Baseline

ResNet18 with lr=0.0002 and no regularization produced textbook *overfitting* : train accuracy reached 90% while validation collapsed to 65%.

#figure(
  image("results/v1/training_curves.png", width: 90%),
  caption: [Version 1: training accuracy climbs while validation stagnates. The red area shows the 28% overfitting gap.],
)

*Lesson*: fine-tuning requires a much lower learning rate than training from scratch.

== Versions 3 to 4 : The One-Line Bug and the Breakthrough

Version 3 introduced a much richer pipeline: EfficientNet-B0, freeze/unfreeze, face extraction, mixup, label smoothing, and warmup. But a `return` inside a loop instead of after it caused only *one parameter* to unfreeze:

```python
# Bug: return INSIDE the loop    |  # Fix: return AFTER the loop
for p in model.parameters():     |  for p in model.parameters():
    p.requires_grad = True        |      p.requires_grad = True
    return  # exits immediately!  |  return  # all params unfrozen
```

Version 3 showed train acc (68%) *below* val acc (73%) for all 50 epochs. The backbone was frozen the whole time, and the model could only learn with its tiny 2-layer classifier head. This single indentation error cost *20 hours of GPU*.

Fixing this one line in Version 4 produced the project's most significant result. At epoch 6 (unfreeze), validation accuracy *jumped from 67% to 77% in a single epoch* as the backbone suddenly started learning deepfake-specific features. The model continued climbing to *89.71%* at epoch 23, with a train/val gap of only 2%, in stark contrast with Version 1's 28% gap.

#figure(
  image("results/v4/training_curves.png", width: 90%),
  caption: [Version 4: the unfreeze at epoch 6 triggers a dramatic drop in both losses. Train and val stay close together, showing healthy generalization.],
)

The resulting *AUC of 0.9671* is a remarkable result: it *exceeds the DeepfakeBench reference of 0.9567 for EfficientNet-B4*, a model with 3.6$times$ more parameters than our B0. This demonstrates that a well-tuned small model with the right training strategy (freeze/unfreeze, adaptive scheduler, proper regularization) can outperform a larger model trained with a standard pipeline.

== Versions 5 to 6 : Scaling to EfficientNet-B4

With EfficientNet-B0 (5.3M parameters) reaching its capacity ceiling at ~90%, the next step was *EfficientNet-B4* (19M parameters) with *pre-extracted faces* for faster training. This phase exposed two distinct pitfalls of working with larger models.

*Version 5 : The configuration trap.* Despite 3.6$times$ more parameters, Version 5 *regressed* to 88.8% test accuracy. Post-hoc analysis found four misconfigurations: wrong resolution (299 instead of B4's native 380), missing BN freeze flag in the SLURM script, no class weighting, and augmentations too aggressive. A larger model amplifies every configuration error.

*Version 6 : Getting it right.* After fixing all V5 issues and aligning with the original paper's choices (native 380$times$380 resolution, class weights, face margin 1.3$times$, BN freeze), the model reached *92.08% validation accuracy*, our best result across all versions. An initial attempt with differential learning rates (backbone at 1/10th the classifier rate) caused training instability, but switching to a uniform learning rate of 3e-05 (proven stable in V4) resolved the issue.

#figure(
  image("results/v6/training_curves.png", width: 90%),
  caption: [Version 6 training curves. The unfreeze triggers rapid improvement. The model peaks at 92% validation accuracy.],
)

= Results

== Global Comparison

#figure(
  table(
    columns: 7,
    align: (left, center, center, center, center, center, center),
    [*Version*], [*Model*], [*Params*], [*Val Acc*], [*Test Acc*], [*AUC*], [*F1*],
    [V1], [ResNet18], [11M], [65.3%], [--], [--], [--],
    [V3 (buggy)], [EffNet-B0], [5.3M], [73.8%], [71.6%], [0.720], [0.817],
    [V4 (fixed)], [EffNet-B0], [5.3M], [89.7%], [*89.8%*], [*0.967*], [*0.920*],
    [V5], [EffNet-B4], [19M], [90.9%], [88.8%], [0.939], [0.917],
    [V6], [EffNet-B4], [19M], [*92.1%*], [89.4%], [0.959], [0.917],
  ),
  caption: [Results across all training versions. V4 remains the best on test set AUC (0.967) despite using a 3.6$times$ smaller model than V6.],
)

== AUC per Manipulation Method

#figure(
  table(
    columns: 6,
    align: (left, center, center, center, center, center),
    [*Method*], [*V3*], [*V4*], [*V5*], [*V6*], [*DeepfakeBench*],
    [Deepfakes], [0.761], [*0.980*], [0.969], [0.973], [0.976],
    [Face2Face], [0.722], [*0.971*], [0.944], [0.970], [0.976],
    [FaceSwap], [0.685], [*0.968*], [0.929], [0.955], [0.980],
    [NeuralTextures], [0.730], [*0.946*], [0.916], [0.937], [0.931],
    [*Global*], [0.720], [*0.967*], [0.939], [0.959], [0.957],
  ),
  caption: [AUC per manipulation method. V4 (EfficientNet-B0) beats DeepfakeBench (EfficientNet-B4) on global AUC despite having 3.6$times$ fewer parameters. NeuralTextures is consistently the hardest method to detect.],
)

== ROC Curves and Confusion Matrix (Best Evaluated Model: V4)

#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  figure(
    image("results/v4/roc_curves_per_method.png", width: 100%),
    caption: [V4 ROC curves per method (AUC=0.967).],
  ),
  figure(
    image("results/v4/confusion_matrix.png", width: 100%),
    caption: [V4 confusion matrix: 97 FP, 332 FN.],
  ),
)

== Why Some Methods Are Harder to Detect

The four manipulation methods in FaceForensics++ produce fundamentally different types of artifacts, which explains the consistent performance gap across all our models:

#figure(
  table(
    columns: 4,
    align: (left, center, center, left),
    [*Method*], [*V4 AUC*], [*V6 AUC*], [*Why this difficulty level*],
    [Deepfakes], [0.980], [0.973], [Autoencoder reconstruction leaves visible blending boundaries and resolution mismatches at the face border],
    [Face2Face], [0.971], [0.970], [Expression transfer creates temporal flickering and jaw boundary artifacts, but preserves identity],
    [FaceSwap], [0.968], [0.955], [3D model fitting produces smoother results; artifacts are mainly lighting and skin tone mismatches],
    [NeuralTextures], [0.946], [0.937], [Only modifies mouth region textures via neural rendering. Minimal spatial artifacts, hardest to distinguish from real],
  ),
  caption: [AUC per method with explanation. Deepfakes leaves the most visible traces (full face replacement); NeuralTextures is the subtlest (localized texture modification).],
)

The images below illustrate the subtlety of these manipulations. Even on the same person, the differences between original and manipulated frames are barely visible:

#figure(
  grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 6pt,
    image("images/ex_original.png", width: 100%),
    image("images/ex_deepfakes.png", width: 100%),
    image("images/ex_neuraltextures.png", width: 100%),
  ),
  caption: [Same person: Original (left), Deepfakes (center, notice the different face), NeuralTextures (right, nearly identical, only mouth region modified).],
)

*Deepfakes* and *FaceSwap* replace the entire face, creating detectable boundaries where the synthetic face meets the original background. *Face2Face* transfers expressions but keeps the original identity, producing artifacts mainly around the jaw and mouth. *NeuralTextures* is the hardest because it only modifies a small region (typically the mouth area) using learned texture synthesis, leaving almost no spatial artifacts. The detector must rely on subtle statistical differences in pixel distributions rather than visible seams.

This difficulty ranking is consistent across all our versions and matches both the original paper and DeepfakeBench, confirming it reflects intrinsic properties of the manipulation methods.

== Example Predictions

#figure(
  image("results/v6/example_predictions.png", width: 80%),
  caption: [Version 6 predictions. Top: correct predictions with high confidence. Bottom: errors. The false negative at 0.93 confidence is the most dangerous case (fake classified as real with high certainty).],
)

= Analysis and Lessons Learned

The iterative process revealed three distinct failure patterns, each diagnosed through the relationship between training and validation metrics:

#figure(
  table(
    columns: 4,
    align: (left, left, left, left),
    [*Pattern*], [*Symptom*], [*Cause*], [*Fix*],
    [Overfitting (V1)], [Train $arrow.t$, Val $arrow.b$], [LR too high, no regularization], [Lower LR, add dropout/mixup],
    [Underfitting (V3)], [Train < Val], [Backbone frozen (bug)], [Assertion to verify unfreeze],
    [Forgetting (V6)], [Both $arrow.b$ after peak], [Backbone-classifier LR mismatch], [Uniform LR],
  ),
  caption: [Three failure patterns discovered across iterations, each diagnosed from the train/val relationship.],
)

*A one-line bug can waste 20 hours of GPU.* The V3 indentation bug froze the backbone entirely. A three-line assertion would have caught it instantly.
*Bigger models need more careful tuning.* B4 (19M params) underperformed B0 (5.3M) in V5 due to four misconfigurations. More parameters means more to corrupt.
*Simplicity beats sophistication.* Uniform lr=3e-05 (V4) outperformed differential LR (V6). Adding complexity without isolated testing introduces risks that other improvements cannot compensate.
*The scheduler matters.* CosineAnnealingLR (V3) decayed too early; ReduceLROnPlateau (V4+) adapted to the model's actual pace.

= Conclusion

Starting from the FaceForensics++ codebase (which provided model architectures and data splits but no training pipeline), a complete deepfake detection system was built and iteratively refined over six training versions. The pipeline progressed from 65% validation accuracy (V1, ResNet18) to 92% (V6, EfficientNet-B4), with the best evaluated model (V4, EfficientNet-B0) achieving 0.9671 AUC, surpassing the DeepfakeBench reference of 0.9567 for EfficientNet-B4 despite using a 3.6$times$ smaller model.

Each version contributed a concrete lesson: V1 on learning rate sensitivity, V3 on the importance of unit testing critical operations, V4 on adaptive scheduling, V5 on the gap between code and configuration, and V6 on the importance of aligning with the reference paper's configuration choices while keeping the training dynamics simple.

*Perspectives.* Three directions remain for future work: (1) obtaining a working XceptionNet checkpoint (the original hosting server's SSL certificate has expired) to compare with the paper's architecture directly; (2) ensemble methods combining B0 and B4 predictions; and (3) cross-dataset generalization, testing whether the detector trained on FF++ c23 transfers to other deepfake datasets (Celeb-DF, DFDC).

== AI Tools

All architectural decisions, hyperparameter choices, experimental analysis, and report writing were carried out by me. AI (Claude) was used as a coding assistant for implementing the training pipeline, debugging SLURM scripts, and proofreading the report for grammar and clarity.

= References

#set text(size: 9.5pt)

+ Rossler et al. "FaceForensics++: Learning to Detect Manipulated Facial Images." _ICCV_ 2019. #link("https://arxiv.org/abs/1901.08971")
+ Yan et al. "DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection." _NeurIPS_ 2023. #link("https://arxiv.org/abs/2307.01426")
+ Tan, Le. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." _ICML_ 2019. #link("https://arxiv.org/abs/1905.11946")
