# An Ensemble Generative Framework for Fault Diagnosis in Data-Limited Photovoltaic Systems - Related Code

## Introduction

This repository contains the code implementation for the research paper titled "An Ensemble Generative Framework for Fault Diagnosis in Data-Limited Photovoltaic Systems". The project addresses the critical challenge of **representative fault signature scarcity** in photovoltaic (PV) diagnostic systems due to constraints like time, equipment damage risk, and safety concerns, leading to data-limited scenarios.

We propose the **Ensemble Particle Swarm Optimization Generative Adversarial Network (EPSOGAN)**, a novel framework that generates high-fidelity synthetic PV fault data. EPSOGAN synergistically integrates multiple GAN variants (WGAN-GP, CTGAN, DCGAN, LSGAN, infoGAN) and optimizes their collective output using Multi-Objective Particle Swarm Optimization (MOPSO) based on complementary evaluation metrics. The goal is to enhance diagnostic model performance, especially when real fault data is scarce.

## Key Features

*   **Novel EPSOGAN Framework:** Integrates five complementary GAN variants (WGAN-GP, CTGAN, DCGAN, LSGAN, infoGAN) to leverage their individual strengths and overcome limitations in modeling complex PV fault signatures under data constraints.
*   **Dynamic Ensemble Optimization:** Employs an innovative Multi-Objective Particle Swarm Optimization (MOPSO) algorithm to dynamically determine the optimal contribution weights for each GAN based on four metrics (MMD, Coverage, Wasserstein Distance, Mode Score), eliminating manual tuning.
*   **Advanced Data Processing:** Incorporates I-V curve correction to normalize environmental effects (temperature, irradiance) and Uniform Manifold Approximation and Projection (UMAP) for effective feature extraction and dimensionality reduction.
*   **Robustness to Data Scarcity:** Specifically designed to resist mode collapse and overfitting, common issues when training individual GANs on limited data.
*   **Validated Performance:** Achieves high diagnostic accuracy (96-99% reported in the paper) even with severely limited training samples (e.g., 7-9 samples for complex faults), demonstrating practical utility.

## Methodology Overview

1.  **Data Acquisition & Processing:** Collect PV operational data (simulated or real), apply I-V curve correction (Eq. \ref{eq:I2}-\ref{eq:Rs1_prime} in the paper) to standardize conditions, and use UMAP for feature extraction (\figref{FIG:feature}).
2.  **Individual GAN Training:** Train the five constituent GAN models (WGAN-GP, CTGAN, DCGAN, LSGAN, infoGAN) independently on the processed fault data.
3.  **MOPSO-based Optimization:** Utilize the MOPSO algorithm (Algorithm \ref{Algorithms:MOPSO_multiple}) with the four evaluation metrics (MMD, Coverage, WD, MS - Eq. \ref{eq:MMD}-\ref{eq:Wasserstein}) as objectives to find Pareto-optimal weight combinations ($\mathbf{w}$) for the GAN ensemble for each specific fault type.
4.  **Ensemble Data Generation:** Generate synthetic fault data by sampling from the individual GANs according to the optimized weights determined by MOPSO (Eq. \ref{eq:score}).
5.  **Evaluation:** Assess the quality of the generated data using the evaluation metrics and evaluate its effectiveness by training downstream fault diagnosis models (e.g., CNN, RBF, SVM, DT, NBM) and testing on real data.

## Code Structure

The main code files include:

*   `data_process.py`: Implements data loading, I-V curve correction based on environmental parameters (temperature, irradiance), and feature extraction/dimensionality reduction using UMAP.
*   `model.py`: Defines the architectures for the five individual GAN variants (WGAN-GP, CTGAN, DCGAN, LSGAN, infoGAN) and potentially the structure for the ensemble generator.
*   `train.py`: Contains the main script for training the individual GAN models and implementing the EPSOGAN framework, including the MOPSO optimization loop (Algorithm \ref{Algorithms:MOPSO_multiple}) to find optimal ensemble weights.
*   `utils.py`: Provides utility functions, potentially including implementations for the evaluation metrics (MMD, Coverage, WD, MS), MOPSO helper functions, data handling, and other shared functionalities.

## Datasets Used in Paper

The paper evaluates the methodology using three dataset configurations to simulate different data availability scenarios:
*   **ISD (Imbalanced Sample Dataset):** Represents realistic scenarios with significant class imbalance (e.g., 7-9 samples for complex faults like DOA+SC/DOA+OC).
*   **BSD (Balanced Sample Dataset):** Represents a data-limited scenario with an equal but small number of samples (30) per fault class.
*   **LD (Large Dataset):** Used as a reference with ample data (1000 samples per class).

*This repository focuses on the EPSOGAN methodology implementation, particularly effective for ISD and BSD scenarios.*

## Current Status and Data Description

**Please Note:**

*   The code provided here implements the EPSOGAN methodology as described in the research paper.
*   Due to laboratory confidentiality agreements and related company interests, the specific **PV fault dataset** used in the research paper is currently **not publicly available**. You would need a dataset with a similar structure (I-V curve data or extracted features for different fault types) to run this code.
*   We plan to consider open-sourcing relevant datasets and code in the future when conditions permit.

## Usage Instructions

(As the specific dataset is not public, running this code directly to reproduce paper results requires a compatible dataset.)

General workflow:

1.  **Prepare Data:** Ensure your PV fault data (after potential I-V curve measurements) is preprocessed into a format suitable for the `data_process.py` script (e.g., CSV files with features and labels). The script expects features amenable to UMAP and subsequent GAN training.
2.  **Configure:** Adjust parameters in configuration files (e.g., `config.json` if used) or script arguments, such as dataset paths, model hyperparameters, MOPSO settings (population size, generations), and training epochs.
3.  **Run Training:** Execute the main training script. This will likely involve training individual GANs followed by the MOPSO optimization to determine ensemble weights and generate the final synthetic dataset.

```bash
# Example run command (modify based on actual script arguments)
python train.py --data_path /path/to/your/processed_data --config config.json --output_dir /path/to/save/results
```
4.  **Evaluate:** Use the generated synthetic data to train downstream diagnostic models (CNN, SVM, etc.) and evaluate their performance on a separate test set of real data.

**Dependencies:** (List major libraries based on paper/code)
*   Python 3.x
*   PyTorch or TensorFlow (depending on implementation)
*   NumPy
*   Pandas
*   Scikit-learn (for SVM, DT, metrics)
*   UMAP-learn (`umap-learn`)
*   Possibly specific libraries for MOPSO or GAN implementations (e.g., `platypus-opt` for MOPSO, specific GAN libraries if used).

*Please refer to the paper for detailed hyperparameters and experimental setup.*

## Evaluation Metrics

The quality of generated data is assessed using:
*   Maximum Mean Discrepancy (MMD)
*   Coverage (C)
*   Wasserstein Distance (WD)
*   Mode Score (MS)

Downstream task performance is evaluated using Accuracy, Recall, and F1-Score on various classifiers. See Tables \ref{tbl:isd_comprehensive} and \ref{tbl:bsd_comprehensive} in the paper for detailed results.

## Contact

If you have any questions regarding the methodology or code implementation (excluding data requests), you can contact us via contact@example.com.
