# Handwritten-Digit-Classification-Using-Neural-Network-MNIST-Dataset-
Introduction
This project focuses on classifying handwritten digits from the MNIST dataset using a Fully Connected Neural Network (FCNN). The dataset contains 60,000 training images and 10,000 test images, each representing a digit from 0 to 9. The goal is to train a neural network to recognize and classify handwritten digits accurately.
Dataset and Preprocessing
•	MNIST Dataset: Grayscale images of size 28x28 pixels representing digits (0-9).
•	Normalization: Pixel values are scaled to the 0-1 range to enhance model performance.
•	One-Hot Encoding: Labels are converted into categorical format for classification.
Model Architecture
The neural network consists of:
1.	Flatten Layer: Converts 28x28 pixel matrix into a 1D array.
2.	Dense Layer (128 neurons, ReLU activation): First hidden layer.
3.	Dense Layer (64 neurons, ReLU activation): Second hidden layer.
4.	Output Layer (10 neurons, Softmax activation): Predicts one of the 10 digit classes.
Training and Optimization
•	Optimizer: Stochastic Gradient Descent (SGD) for minimizing loss.
•	Loss Function: Categorical Cross-Entropy for multi-class classification.
•	The model is trained for 10 epochs with a batch size of 32.
Evaluation and Results
•	The model achieves a final test accuracy displayed after training.
•	Plots for training vs. validation loss and training vs. validation accuracy are generated to visualize performance trends.
Prediction on New Images
•	A function is implemented to visualize and predict a single digit from the test dataset.
Conclusion
This project demonstrates an effective FCNN for digit recognition. The accuracy can be further improved by implementing convolutional neural networks (CNNs), experimenting with different optimizers, or adding dropout layers to prevent overfitting.

