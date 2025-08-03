# Ayna ML Assignment: Conditional U-Net for Polygon Coloring

## Project Overview

This project addresses the Ayna ML intern assignment, which involves training a model to generate an image of a colored polygon. The model takes two inputs: an image of a polygon outline and a desired color name (e.g., "yellow"). It outputs an image of the polygon filled with the specified color.

The solution involves implementing a **Conditional U-Net** from scratch in PyTorch. The project was tracked using Weights & Biases, and this report details the architecture, training process, final results, and key learnings.

**Final Result Preview:**
| Input | Target | Prediction |
| :---: | :---: | :---: |
| ![star_outline](https://i.imgur.com/gYfM7t2.png) | ![star_yellow](https://i.imgur.com/i9Jz3tM.png) | ![star_pred](https://i.imgur.com/eBwG8yE.png) |
| *Star Outline* | *Target (Yellow)* | *Model Prediction (Yellow)* |

*(Note: You can replace these placeholder image links with screenshots from your final output.)*

---

## Table of Contents
1.  [Model Architecture](#1-model-architecture)
2.  [Hyperparameters & Training Setup](#2-hyperparameters--training-setup)
3.  [Training Dynamics & Results](#3-training-dynamics--results)
4.  [Key Learnings](#4-key-learnings)
5.  [How to Run](#5-how-to-run)
6.  [W&B Project](#6-wb-project)

---

## 1. Model Architecture

The core of this project is a **U-Net**, which is an encoder-decoder architecture with skip connections, making it highly effective for image-to-image translation tasks where spatial information must be preserved.

### Color Conditioning Mechanism

To make the U-Net "conditional," the color name input is integrated into the model's architecture. This was achieved through the following steps:
1.  **Color Embedding:** A global mapping of all unique color names to integer indices was created. This integer is fed into an `nn.Embedding` layer, converting it into a dense vector representation.
2.  **Projection:** The embedding vector is passed through a linear layer (`nn.Linear`) to project it to the same channel dimension as the U-Net's bottleneck.
3.  **Injection:** The projected color vector is then reshaped and **added** to the bottleneck feature map of the U-Net. This provides the entire decoder path with the global context of the desired output color, allowing it to accurately fill the polygon shape.

---

## 2. Hyperparameters & Training Setup

The final model was trained with the following configuration:

| Hyperparameter | Value |
| :--- | :--- |
| **Framework** | PyTorch |
| **Optimizer** | Adam |
| **Loss Function**| L1 Loss (`nn.L1Loss`) |
| **Learning Rate** | `5e-4` |
| **LR Scheduler** | ReduceLROnPlateau |
| **Weight Decay** | `1e-5` |
| **Epochs** | 200 |
| **Batch Size** | 8 |

### Rationale for Key Choices
* **Loss Function**: The project initially started with Mean Squared Error (`MSELoss`). However, this resulted in blurry outputs with color bleeding. I switched to **Mean Absolute Error (`L1Loss`)**, which is known to be less sensitive to large errors and encourages the model to produce sharper, more defined edges. This change was critical for achieving high-quality visual results.

---

## 3. Training Dynamics & Results

The training process involved several iterations to diagnose and fix issues, demonstrating a methodical approach to model development.

### Iteration 1: Fixing Blurry Outputs

* **Problem:** The initial model trained with `MSELoss` produced blurry images with hazy backgrounds.
* **Solution:** Changed the loss function to `L1Loss`. This immediately improved edge sharpness and background clarity.

| Before (MSE Loss) | After (L1 Loss) |
| :---: | :---: |
| *[Insert Image of Blurry Output Here]* | *[Insert Image with Sharper Edges but Wrong Color Here]* |

### Iteration 2: Solving Incorrect Color Mapping

* **Problem:** After fixing the blurriness, a more subtle bug appeared: the model consistently mapped certain colors incorrectly (e.g., an input of "yellow" produced a "magenta" output).
* **Solution:** This was traced back to an inconsistent mapping of color names to indices between the training and validation datasets. Each dataset was creating its own `color_to_idx` map. The bug was fixed by implementing a **single, global color map** created from the union of all colors in both `train.json` and `val.json`. This ensured that "yellow" (and every other color) had the same index throughout the entire training and validation process.

### Final Results

After implementing these fixes, the model trained successfully and converged well.

* **Loss Curves:** The training and validation loss curves tracked each other closely over 200 epochs, indicating that the model generalized well without overfitting.
    * *[Insert your final Training/Validation Loss Curve screenshot here]*
* **Quantitative Metrics:** The final model achieved excellent performance on the validation set:
    * **Validation MAE:** **0.0071**
    * **Validation PSNR:** **32.32 dB**
* **Qualitative Results:** The model now produces high-quality, accurate images for all colors in the dataset.
    * *[Insert a grid of your final, successful predictions here]*

---

## 4. Key Learnings

This project provided several important insights into building generative models:
1.  **Loss Functions Matter:** The choice of loss function has a profound and direct impact on the visual quality of the output. For image generation, `L1Loss` is often superior to `MSELoss` for preserving sharpness.
2.  **Data Pipeline is Critical:** The most challenging bug (color swapping) was not in the model architecture but in the data pipeline. Ensuring consistent data representation and preprocessing (like the global color map) is paramount for stable training.
3.  **Iterative Debugging Works:** Successful model development requires a methodical process of identifying a problem, forming a hypothesis, implementing a fix, and re-evaluating. Each step in this project's evolution was a direct result of this loop.

---

## 5. How to Run

### a) Training

The full, final training script is provided as `train_model.py` (or your chosen filename). To retrain the model, ensure the dataset is in the correct path and run:
```bash
python train_model.py