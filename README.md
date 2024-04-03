# Disease Prediction system  from symptoms  using Machine Learning

## Project Overview
This project focuses on building a disease prediction system using machine learning algorithms. The system aims to aid in the early detection of diseases by analyzing input symptoms. It employs a combination of supervised and unsupervised learning techniques to identify patterns in symptom-disease relationships. Few dimensionality reduction techniques also used to make the model more accurate.The technology stack includes Flask for web development, Python for backend processing, and Matplotlib for data visualization.

## Explored Algorithms
Various machine learning algorithms have been explored for disease prediction, including:

-Regression Models
- Decision Tree
- Naive Bayes
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Probabilistic Models

## Performance Metrics Utilization
Performance metrics such as accuracy, precision, recall, and F1-score have been utilized to evaluate the effectiveness of the disease prediction model. These metrics provide insights into the model's ability to correctly classify diseases based on input symptoms. Additionally, techniques such as cross-validation and confusion matrix analysis have been employed to further assess the model's performance and identify areas for improvement.

## User Interface
The system features an intuitive user interface for symptom input and displays the top 3 predicted diseases. Random Forest has been chosen as the primary algorithm due to its superior performance during evaluation(compared every algo and using performance metrics random forest provides best results with good accurarcy score).

## Usage
To utilize the system, ensure that all dependencies located in the `app` folder are installed using the following commands:

<details>
<summary>Installation Commands</summary>

```bash
pip install -r requirements.txt
python app.py
```
</details>
<details>
<summary>Import Statements</summary>
  
```bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from flask import Flask, request, jsonify, render_template
```
</details>


**Note:** This project is intended for educational purposes only.

## Project Report
For more detailed information, refer to the full project report available [here](DISEASE_PRED.pdf).



