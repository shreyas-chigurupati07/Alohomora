# RBE/CS 549 Computer Vision: Alohomora Project

## Project Overview

This project is divided into two main parts: "Shake my Boundary," which explores probability-based edge detection using various parameters and filters, and "Deep Dive into Deep Learning," where multiple deep learning architectures are compared for object classification on the CIFAR10 and BSDS500 datasets.

## Key Components

1. **Probability-Based Edge Detection:** Implements edge detection by considering texture, brightness, and color variations in the image, utilizing filters like Oriented Derivative of Gaussian, Leung-Malik, and Gabor Filter-banks.
2. **Deep Learning Model Comparison:** Evaluates and compares the performance of various deep learning architectures, including ResNet, DenseNet, and ResNeXt, on the CIFAR10 dataset to classify objects accurately.

## Implementation Details

- Edge Detection: Utilizes a combination of texture maps, brightness maps, color maps, and half-disc masks to detect boundaries in images effectively.
- Deep Learning: Trains and tests multiple architectures with a focus on training accuracy, loss values, and computational efficiency, employing techniques like normalization, standardization, and various data augmentations.

## Results

The project successfully enhances boundary detection using a novel, probability-based approach and provides an in-depth analysis of deep learning models, demonstrating their strengths and weaknesses in object classification tasks.
[Color_1](https://github.com/shreyas-chigurupati07/Alohomora/assets/84034817/fd7b86f9-f6ab-48cb-bf4e-d0f38682d315)
!![Texton_1](https://github.com/shreyas-chigurupati07/Alohomora/assets/84034817/e5d73300-4ab4-42cd-b455-178f913b1e16)


## How to Run

1. Clone the repository: `git clone [repository-link]`
2. Navigate to the project directory: `cd [project-directory]`
3. Execute the scripts for edge detection and deep learning model training/testing.

## Dependencies

- Python
- NumPy
- OpenCV
- PyTorch
- torchvision


## References

- Various resources and publications related to edge detection and deep learning model architectures.
