# Computer Vision: Final Project 
### Project 2: Joint Detection of AI-Generated Images and Post-Processing Alterations in Real-World Scenarios
### By Angela Petkova (2288299) and Oliver van Douveren (2285369)

## Introduction
AI-generated image detectors are usually trained and evaluated on pristine, unmodified images straight from generative models — but real-world images (especially on social media) get compressed, re-uploaded, screenshotted, or otherwise post-processed before anyone sees them. These transformations introduce artifacts that can mask or mimic the very signals detectors rely on, so a model that performs well in the lab may fail in the wild.
This project implements a multi-task deep learning model that simultaneously performs two forensic tasks on a single input image: (1) classifying it as real or AI-generated, and (2) identifying which post-processing transformation (original, internet-transmitted, or re-digitalized) was applied.

### Repo usage
On this repo, we contained the code as per Exam Regulations in 

## Definitions
1. Data Preparation

Trained on Google Colab's free tier (NVIDIA Tesla T4, 15GB VRAM, 12.7GB RAM, 113GB disk).
Subset of ~18,000 images (~1.7MB each, ~30.6GB total) selected to fit within Colab's storage while leaving room for checkpoints, logs, and cache.
Family grouping: Images sharing the same source ID (different processed versions of the same original) were grouped together before splitting, to prevent data leakage across train/val/test sets.
Final split: 80% train / 10% validation / 10% test, performed at the family level.
We managed to define two global variables, setting the to resize_image_size = 380 and fixing global_batch_size = 16.

2. Model Architecture
2.1 Dataset & Dataloader
A custom PyTorch Dataset was built for EfficientNet-B4, loading each image alongside its two labels (auth_label and transform_label), enabling joint multi-task batching.
2.2 Multi-Task EfficientNet-B4

Multiple backbones (ResNet, EfficientNet variants) were evaluated; EfficientNet-B4 gave the best accuracy, consistent with prior work (Shafiq & Pratap, 2026) on EfficientNet for AI-image detection.
The original ImageNet classification head was removed, and the backbone was repurposed as a shared feature extractor with two task-specific heads: authenticity classification and transformation classification.

2.3 Baseline Models

Single-task baselines (Authenticator-only and Transformation-only) were built using the same backbone/code structure as the multi-task model, to ensure fair comparison.

2.4 Multi-Task Training

Full end-to-end fine-tuning (not just the new heads), justified by Yosinski et al. (2014)'s finding that deep features become increasingly task-specific in later layers — ImageNet features alone are insufficient for detecting generation/transformation artifacts.
Initial setup: equal loss weights (1.0 / 1.0), Adam optimizer, learning rate 0.0001 (best of tested values), cosine annealing scheduler, weight decay 1e-4.
Checkpointing: best model saved based on combined validation accuracy (not just final epoch).
15 epochs per model, chosen as a compromise given Colab limits (~8 epochs per session → 2 sessions per run), enabling all 5 loss-weight configurations to be trained within scope.
Training function supports both training from scratch and resuming from checkpoints.

3. Ablation Study — Loss Weight Configurations
AUTH_LOSS_WEIGHTTRANS_LOSS_WEIGHTPurpose1.01.0Baseline multitask model0.251.0Reduce importance of real/fake task0.11.0Strongly reduce real/fake task1.00.25Reduce importance of transformation task1.00.1Strongly reduce transformation task
Only loss weights were varied across runs; all other hyperparameters held constant for isolated comparison.



# Discussion

### 1 Performance analysis across transformation types


### 2 Ablation Study
Our hypothesis was that, in the multi-task learning setting, reducing the loss weight assigned to a specific task would weaken that task’s contribution to the overall optimization objective and may therefore lead to lower performance on that task. This expectation is based on the fact that the total objective is formulated as a weighted combination of task-specific losses, meaning that the relative loss weights influence the optimization priority given to each task, as discussed by Kendall et al. (2018). However, we also hypothesized that this relationship would not necessarily be strictly monotonic, since the two tasks may either compete for shared representational capacity or benefit from shared feature learning.

### 3 Lowering the transformation loss

| $w_{\text{AI/original}}$ | $w_{\text{transform}}$ | AI/original Accuracy | Transformation Accuracy |
| -----------------------: | ---------------------: | -------------------: | ----------------------: |
|                     1.00 |                   1.00 |               94.22% |                  97.06% |
|                     1.00 |                   0.25 |               94.06% |                  94.28% |
|                     1.00 |                   0.10 |               94.06% |                  88.61% |

Table 1: Effect of reducing the transformation loss weight while keeping the AI/original loss weight fixed.

The results shown in Table 1 support our hypothesis that reducing the loss weight assigned to a task can weaken its optimization priority and lead to lower performance on that task. In Table 1, decreasing the transformation loss weight from 1.00 to 0.25 and then to 0.10 leads to a clear drop in transformation accuracy, from 97.06% to 94.28% and 88.61%, respectively. Meanwhile, AI/original accuracy remains nearly unchanged, suggesting that reducing the auxiliary transformation loss mainly affects the transformation classification branch rather than substantially improving or degrading the AI/original task.

| $w_{\text{trans}}$ | Worst transformation type | AI/original accuracy |
| -----------------: | ------------------------- | -------------------: |
|               1.00 | redigital                 |               92.67% |
|               0.25 | redigital                 |               93.17% |
|               0.10 | redigital                 |               92.83% |

Table 2: Worst transformation type while reducing the transformation loss weight while keeping the AI/original loss weight fixed.

Table 2 shows that across all loss-weight configurations, re-digitalized images produced the lowest AI/original classification accuracy, suggesting that the re-digitization process introduces degradation patterns that make authenticity detection more difficult.

### 4 Lowering the authenticity

The results shown below in Table 3 do not support the hypothesis. Reducing the AI/original loss weight did not harm AI/original classification.

| $w_{\text{auth}}$ | AI/original Accuracy |
| ----------------: | -------------------: |
|              1.00 |               94.22% |
|              0.25 |               93.72% |
|              0.10 |               94.83% |

Table 3: Effect of reducing the authenticity loss weight while keeping the transformation loss weight fixed.

The improvement at $w_{\text{auth}} = 0.10$ suggests that AI/original classification may benefit from features learned through the transformation task.

| $w_{\text{auth}}$ | Transformation Accuracy |
| ----------------: | ----------------------: |
|              1.00 |                  97.06% |
|              0.25 |                  97.39% |
|              0.10 |                  98.28% |

Table 4: Increase of transformation accuracy when $w_{\text{auth}}$ drops.

Table 4 supports the idea that reducing the competing task's weight allows the model to focus more on transformation classification.

### 5 Conclusion

The ablation study shows that the effect of loss weighting is not strictly monotonic. Lowering the transformation loss weight clearly reduced transformation accuracy, which supports the hypothesis that giving a task less weight can weaken its optimization priority. However, lowering the AI/original loss weight did not reduce AI/original accuracy and even slightly improved it in the lowest-weight setting.

This suggests that AI/original detection benefits from the transformation-related features learned by the shared backbone. Therefore, the hypothesis is only partially supported: loss weighting affects task performance, but the direction and strength of this effect depend on how much the tasks share useful representations.
