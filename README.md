# Parkinson's Disease Analysis

This project is designed to analyze and predict the progression of Parkinson's Disease using biomedical voice measurements. It allows for various data analysis tasks such as comparing trends based on gender, visualizing correlations between variables, predicting patient progression, and more. The datasets used in this project include data from patients with early-stage Parkinson's disease, specifically related to motor and speech features.

## Datasets Overview

### Dataset 1: Parkinson's Disease Speech Data (42 Patients)
This dataset contains biomedical voice measurements from 42 individuals with early-stage Parkinson's Disease, collected as part of a trial for telemonitoring devices. Each row represents a voice recording from one of these 42 patients, with measurements across 16 voice features and the Unified Parkinson’s Disease Rating Scale (UPDRS).

#### Key Parameters:
- **motor_UPDRS**: A score representing the severity of motor symptoms in Parkinson's Disease.
- **total_UPDRS**: The total score combining motor and non-motor aspects of Parkinson’s Disease.
- **Jitter**, **Shimmer**: Measures of voice frequency and amplitude instability, which are useful for detecting symptoms of Parkinson's.
- **HNR**: Harmonics-to-Noise Ratio, indicating voice clarity.
- **RPDE**, **DFA**: Measures of the complexity and scaling properties of the voice signal, indicating the irregularity often seen in Parkinson's Disease.

### Dataset 2: Parkinson's Disease Speech Data (31 Patients)
This dataset is composed of data from 31 patients, 23 with Parkinson's disease (PD) and 8 healthy controls. It includes voice measurements such as fundamental frequency, jitter, shimmer, and more. The goal is to discriminate between healthy individuals and those with Parkinson's disease.

#### Key Parameters:
- **MDVP:Fo(Hz)**: Average fundamental frequency of the voice.
- **Jitter**, **Shimmer**: Various measures of instability in voice frequency and amplitude.
- **status**: Health status of the subject (1 = Parkinson’s Disease, 0 = Healthy).
- **RPDE**, **D2**, **DFA**: Measures related to the nonlinear complexity of the voice signal.

## Features

- **Compare Gender Trends with P-values**: Analyze and compare trends in voice measures between male and female patients.
- **Plot Correlation Matrix**: Visualize the correlations between different voice measures and clinical scores.
- **Predict Patient Progression**: Predict the progression of Parkinson’s disease for a specific patient or the mean patient.
- **Plot Disease Progression**: Visualize the progression of Parkinson’s disease over time.
- **Train and Evaluate Model**: Train machine learning models to detect Parkinson's Disease from voice measurements.
- **Check Datasets Info**: Get an overview of the datasets used in this project.

## Research Questions

### 1. **What differences are there between male and female patients in terms of voice features and Parkinson's progression in the study?**

   **Function Tested**: `compare_gender_trends()`

   The function compares voice features across male and female patients to identify any significant differences, using statistical tests to evaluate if gender is correlated with certain speech patterns or progression in Parkinson’s disease.

### 2. **How strongly are features correlated in Parkinson's Disease?**

   **Function Tested**: `plot_correlation_matrix()`

   This function generates a correlation matrix to visualize the relationships between voice features (like Jitter, Shimmer, HNR) and clinical scores (motor_UPDRS, total_UPDRS). By plotting these correlations, we can determine which features are more strongly correlated with the progression of Parkinson’s disease.

### 3. **Can we predict the progression of Parkinson's Disease (motor_UPDRS and total_UPDRS scores) based on voice features?**

   **Function Tested**: `predict_patient_progression()`, `train_and_evaluate_model()`

   These functions test if it’s possible to predict a patient's motor and total UPDRS scores based on the voice measurements. The model is trained using machine learning techniques to understand and predict disease progression from speech-related features.

### 4. **How does Parkinson's Disease progress over time in a specific patient or a mean patient?**

   **Function Tested**: `plot_disease_progression()`

   This function visualizes the disease progression over time for a specific patient or the average patient. It helps answer how the disease manifests over several recordings and whether there is a pattern in progression.

### 5. **Building a machine learning model to distinguish between healthy individuals and those with Parkinson's Disease using voice data and UPDRS score**

   **Function Tested**: `train_and_evaluate_model()`

   The function trains a machine learning model to classify patients as either healthy or affected by Parkinson’s Disease based on voice features. It is evaluated using performance metrics like accuracy, precision, recall, and F1 score to assess the model's ability to distinguish between these groups.


## Installation

Follow the steps below to set up the project and install the required dependencies.

### 1. Clone the repository
```bash
git clone https://github.com/your-username/parkinsons-disease-analysis.git
cd parkinsons-disease-analysis
```
### 2. Install Runtime and Development Dependencies

We use `pyproject.toml` for managing dependencies. To install the necessary packages for both runtime and development, use the following command:
```bash
pip install .[dev]
```
This command will install the dependencies for the development environment as well as any runtime packages required for the project.

## Usage
To start the project, simply run the main.py file. The GUI will open, allowing you to interact with the analysis tools.

You will be presented with a menu where you can choose different functionalities, including:

* Comparing gender trends
* Plotting correlation matrices
* Predicting patient progression
* Training and evaluating a model