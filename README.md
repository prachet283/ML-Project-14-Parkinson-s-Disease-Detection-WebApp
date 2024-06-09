# ML Parkinson’s Disease Detection

# Introduction
This project aims to build a machine learning model to detect Parkinson’s Disease using various features extracted from voice recordings. Parkinson's Disease is a progressive nervous system disorder that affects movement. Early detection can significantly improve the quality of life for patients.

# Dataset
The dataset used for this project is sourced from the UCI Machine Learning Repository. It contains biomedical voice measurements from 31 people, 23 with Parkinson's disease.

Features:

MDVP
(Hz) - Average vocal fundamental frequency

MDVP
(Hz) - Maximum vocal fundamental frequency

MDVP
(Hz) - Minimum vocal fundamental frequency

etc

Target:
Status - Health status of the subject (1 - Parkinson's, 0 - healthy)

# Technologies Used
Python Streamlit scikit-learn

# Model
The Support Vector model is used to detect the Parkinson based on the provided parameters. The model is trained using the SVM class from the scikit-learn library.

# Results
The model achieves an accuracy of approximately 87.17% on the test set.

# Deployment
The application is deployed using Streamlit. You can access it here = https://ml-project-10-credit-card-fraud-detection-7xxqchevqpmzgdq5i9s6.streamlit.app/

# Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions or improvements.

# Contact
If you have any questions or suggestions, feel free to contact me at prachetpandav283@gmail.com .
