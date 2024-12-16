# CrimeCast-Forecasting-Crime-Categories

Crime Category Prediction Using Machine Learning

## Description

This Jupyter Notebook provides a comprehensive workflow for predicting crime categories based on various features of a dataset. The project includes data preprocessing, feature engineering, and training machine learning models with optimized hyperparameters. It aims to achieve high accuracy in crime classification using techniques such as scaling, encoding, and ensemble models.

## Features

- **Data Preprocessing**: Includes handling missing values, encoding categorical variables, and scaling numerical features.
- **Exploratory Data Analysis (EDA)**: Identifies key trends and distributions in the data.
- **Machine Learning Models**:
  - Decision Tree Classifier
  - XGBoost Classifier (with hyperparameter tuning)
  - Linear SVC
- **Model Evaluation**: Evaluates models using metrics such as accuracy, classification report, and confusion matrix.
- **Hyperparameter Optimization**: Implements RandomizedSearchCV for fine-tuning XGBoost parameters.

## Libraries Used

The notebook utilizes the following Python libraries:

- `numpy` - For numerical computations.
- `pandas` - For data manipulation and analysis.
- `matplotlib` - For data visualization.
- `seaborn` - For statistical plotting.
- `sklearn` - For preprocessing, model training, and evaluation.
- `xgboost` - For gradient-boosted decision trees.
- `lightgbm` - For light gradient-boosted decision trees.
- `joblib` - For saving models and encoders.
- `os` - For file operations.
- `warnings` - To suppress warnings.

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git

2. Navigate to the project directory
   ```bash
   cd your-repository
3. Install the required dependencies:
  ```bash
  pip install -r requirements.txt
```
4. Open the notebook in Jupyter:
    ```bash
    jupyter notebook crime-category-prediction.ipynb

5. Run the cells sequentially to execute the workflow.

### Results
* The notebook demonstrates:

* Data preprocessing to prepare datasets for modeling.
* Model performance metrics such as accuracy, confusion matrices, and classification reports.
* Hyperparameter tuning for improving prediction accuracy.
  
### Acknowledgments
* This project leverages publicly available datasets and Python libraries to demonstrate a real-world machine learning workflow. Thanks to the open-source community for their tools and     datasets.

