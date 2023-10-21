# Intrusion Detection using Machine Learning 🛡️

🔒🚀This repository houses a comprehensive Intrusion Detection System (IDS) project built on machine learning techniques, with a primary focus on the NSL-KDD dataset. Our goal is to develop a robust, scalable, and efficient system to detect and prevent intrusions in computer networks.


## Overview

This project focuses on building an Intrusion Detection System using machine learning models. The primary goal is to safeguard computer networks against various forms of cyber threats and malicious activities.


## Dataset

📂 The NSL-KDD dataset was used for training and evaluating the models. The NSL-KDD dataset is a well-known benchmark dataset for intrusion detection, containing a large number of network traffic instances categorized into normal and attack classes.

Dataset source: [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html)


## Models

🔍 Three machine learning models were implemented in this project:

1. **LSTM (Long Short-Term Memory)**: A recurrent neural network model designed to handle sequential data for intrusion detection.

2. **CNN (Convolutional Neural Network)**: A deep learning model typically used for image processing, adapted to detect intrusions by processing network traffic data.
   
4. **FDNN (Feedforward Neural Network)**: A feedforward neural network is a type of artificial neural network where data flows in one direction, from input layer to output layer, without loops or feedback connections.

- Some basic models where also implemented to see how they perform and vary from the neural networks
  
5. **Gaussian Naive Bayes**: It's a probabilistic classifier based on the Bayes' theorem and assumes that features are conditionally independent. It's simple and efficient, often used for text classification and spam detection.

6. **Logistic Regression**: It models the probability that a given input belongs to one of the two classes using the logistic function. It's a linear model and works well for problems with linearly separable classes.

7. **K-Nearest Neighbors (KNN)**: It classifies data points based on the majority class of their k-nearest neighbors. KNN is a non-parametric and instance-based classifier.

8. **Decision Tree Classifier**: It partitions the feature space into regions and assigns a class label to each region. Decision trees can model complex relationships in the data but are prone to overfitting.

9. **Random Forest Classifier**: It's an ensemble method that combines multiple decision trees to improve prediction accuracy and reduce overfitting. Random Forest is highly versatile and effective for a wide range of binary classification tasks.


## Model Performances

📊 Model performance metrics were calculated to evaluate their effectiveness:

- Accuracy
- Precision
- Recall
- F1 Score
- R2 Score

🔍📊 You can find all the appropirate scores either in the beginning of their respective .ipynb or their cell outputs.


## Code Structure

📁 The repository structure is organized as follows:

- `LSTM.ipynb`: Jupyter Notebook containing the code for the LSTM model.
- `CNN.ipynb`: Jupyter Notebook containing the code for the CNN model.
- `Binary_Classification+FNN`:  Jupyter Notebook containing the codes for 5 basic classifiers and Feedforward Neural Network model.


## Usage

🚀 Follow these steps to use the repository:

1. Ensure you have the necessary libraries and dependencies installed.
2. Open the relevant Jupyter Notebook (`LSTM.ipynb` or `CNN.ipynb`) to train and evaluate the models.
3. Follow the instructions in the notebook to load the dataset and run the code.


## Setup

🛠️ To set up this repository, you can follow these steps:

1. Clone the repository to your local machine:

   ```bash
   https://github.com/Maatrika-P/Intrusion-Detection-System.git

2. Install the required dependencies:
   
    ```bash
    pip install -r requirements.txt
    
3. Open the Jupyter Notebooks and start experimenting with the models.


## Conclusion
📈 This project showcases the application of machine learning models for intrusion detection using the NSL-KDD dataset. The choice between LSTM and CNN models depends on the specific requirements and characteristics of the network traffic data.

Feel free to explore and modify the code to suit your needs and improve the detection accuracy.


## Acknowledgments
🙏 Special thanks to the contributors of the NSL-KDD dataset for providing valuable data for research and development.


## Contribution
Contributions are welcome! Whether you want to suggest improvements, submit bug reports, or add new features, feel free to open an issue or create a pull request.

Join us in building an intelligent defense system against network intrusions with machine learning. 🛠️🖥️💼

💼 Credits to the developers and maintainers of machine learning and deep learning libraries used in this project.
