
# MNIST Classification Using a Fully Connected Neural Network

![image](https://github.com/user-attachments/assets/fd16d9fa-0875-4875-bdf5-2fb821dd4332)

## Table of contents

1. [Overview](#overview)
2. [Network Architecture](#network-architecture)
3. [Hyperparameters](#hyperparameters)
4. [Results](#results)
   1. [Without BatchNorm](#without-batchnorm)
   2. [With BatchNorm](#with-batchnorm)
5. [Analysis & Observations](#analysis--observations)
6. [Regularization & Improvements](#regularization--improvements)
7. [Requirements](#requirements)
8. [Authors](#authors)


## Overview

This project implements a fully connected Deep Neural Network (DNN) from scratch using PyTorch to classify the MNIST dataset. The goal was to examine the impact of Batch Normalization (BatchNorm) on training speed and accuracy. Two configurations were tested:

- Without BatchNorm

- With BatchNorm

The results indicate that BatchNorm did not improve performance in this case, providing an interesting insight into its effectiveness under specific configurations.
## Network Architecture

The model consists of the following layers:

- #### Input Layer: 784 neurons (28x28 grayscale images flattened)

- #### Hidden Layers:

    - 1st hidden layer: 20 neurons

    - 2nd hidden layer: 7 neurons

    - 3rd hidden layer: 5 neurons

    - Output Layer: 10 neurons (corresponding to the 10 MNIST classes)

#### Activation Functions:

- ReLU for hidden layers

- Softmax for output layer

#### Loss Function:

- Cross-entropy loss
## Hyperparameters

- **Learning rate:** 0.009

- **Batch size:** 256

- **Maximum epochs:** 200

- **Stopping criterion:** Early stopping after 100 consecutive iterations without improvement in validation loss.
## Results

## Without BatchNorm
- **Training epochs:** 58

- **Total training time:** 120.75 seconds

- **Batches to convergence:** 11,230

- **Final Accuracies:**

    - Train accuracy: 91.89%

    - Validation accuracy: 91.53%

    - Test accuracy: 91.57%
 
  ![image](https://github.com/user-attachments/assets/fc519fe9-cea5-4ab1-bba6-4ae2b4f94464)

## With BatchNorm
- **Training epochs:** 64

- **Total training time:** 197.44 seconds

- **Batches to convergence:** 12,357

- **Final Accuracies:** 

    - Train accuracy: 88.44%

    - Validation accuracy: 88.3%

    - Test accuracy: 88.34%

![image](https://github.com/user-attachments/assets/bf5a6a88-fdeb-4422-ae84-9bee650bd02c)

## Analysis & Observations

- **BatchNorm did not improve classification accuracy** in this specific case. The network without BatchNorm achieved a higher test accuracy (91.57%) compared to with BatchNorm (88.34%).

- **Training Speed:** The model trained faster without BatchNorm (58 epochs vs. 64 epochs).

- **Numerical Stability:** Contrary to expectations, BatchNorm did not improve numerical stability in this case. This could be due to applying BatchNorm before activation instead of after, which is often recommended for better performance.
## Regularization & Improvements

## L2 Regularization

I introduced L2 Regularization to improve generalization:

- compute_cost Function:

    - Added an L2 regularization term: (λ/2m) * sum(W^2) to penalize large weights.

    - The final cost now includes both cross-entropy loss and L2 regularization.

- update_parameters Function:

    - Adjusted weight updates to include λ * W in gradient computations.

    - L2 regularization is optional and controlled via a parameter.

- **Impact of L2 Regularization:**

    - Helped reduce overfitting by shrinking model weights towards zero.

    - Led to more stable training with improved generalization.

![image](https://github.com/user-attachments/assets/9a2e931f-ddbb-452c-a767-2205fabe0fdb)

## Requirements

- Python 3.8+

- PyTorch

- NumPy

- Jupyter Notebook
## Authors

- Roi Garber

- Nicole Kaplan
