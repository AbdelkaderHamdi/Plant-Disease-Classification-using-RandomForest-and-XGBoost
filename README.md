# Plant Disease Classification using RandomForest and XGBoost

## Project Overview

This project focuses on classifying plant diseases using machine learning models, specifically RandomForest and XGBoost. It is a refactored version of an existing notebook that originally utilized PyTorch/TensorFlow for deep learning-based classification. The primary goal of this refactoring was to explore the performance of traditional ensemble methods and potentially improve accuracy for plant disease classification tasks.

## Key Features

- **Ensemble Learning:** Implements and compares two powerful ensemble learning algorithms: RandomForest Classifier and XGBoost Classifier.

- **Hyperparameter Tuning:** Utilizes `RandomizedSearchCV` to efficiently find optimal hyperparameters for both RandomForest and XGBoost models, ensuring robust performance.

- **Comprehensive Evaluation:** Provides detailed classification metrics including accuracy score, classification report (precision, recall, f1-score), and confusion matrices for a thorough assessment of model performance.

- **Feature Importance Analysis:** Visualizes and reports the most important features identified by the trained models, offering insights into which image characteristics are most influential in disease detection.

- **Structured Notebook:** Maintains the original well-structured architecture of the notebook, with clear cell groupings, titles, and a logical data pipeline.

- **Data Preprocessing:** Includes functions for loading, splitting, and preprocessing image data to prepare it for traditional machine learning algorithms.

## Getting Started

### Prerequisites

To run this notebook, you will need Python 3.x and the following libraries:

- `numpy`

- `pandas`

- `matplotlib`

- `seaborn`

- `opencv-python` (for `cv2`)

- `scikit-learn`

- `xgboost`

- `jupyter` (to run the .ipynb notebook)

You can install these using pip:

```bash
pip install numpy pandas matplotlib seaborn opencv-python scikit-learn xgboost jupyter
```

### Data

This project requires a dataset of plant images categorized by disease. The original notebook was designed to work with a directory structure where each subdirectory represents a different plant disease class. You will need to replace `/path/to/your/plant-disease-dataset` in the notebook with the actual path to your dataset.

Example directory structure:

```
plant-disease-dataset/
├── class_A/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class_B/
│   ├── imageX.jpg
│   ├── imageY.jpg
│   └── ...
└── ...
```

### Running the Notebook

1. **Download the notebook:** Save the `plant-disease-detection-project-final.ipynb` file to your local machine.

1. **Place your dataset:** Ensure your plant disease image dataset is organized as described above and update the `data_dir` variable in the notebook to point to its location.

1. **Launch Jupyter Notebook:**

1. **Open the notebook:** Navigate to `plant-disease-detection-project-final.ipynb` in the Jupyter interface and open it.

1. **Run all cells:** Execute all cells in the notebook sequentially. The notebook will handle data loading, preprocessing, model training (with hyperparameter tuning), evaluation, and visualization of results.

## Project Structure

The notebook is organized into logical sections, each with clear markdown headings and comments:

- **1| Importing some modules:** Essential libraries for data handling, visualization, and machine learning.

- **2| Creating some functions and create dataframe:** Functions for defining data paths, creating dataframes, and splitting data into training, validation, and test sets.

- **3| Function to preprocess images for Machine Learning Models:** Handles image loading, resizing, normalization, and flattening for use with RandomForest and XGBoost.

- **4| Function to display data sample (adapted for traditional ML):** Utility to visualize sample images from the dataset.

- **5| Data Loading and Preparation:** Loads the dataset and prepares it for model training.

- **6| Image Preprocessing for Machine Learning Models:** Executes the image preprocessing steps.

- **7| RandomForest Classifier with Hyperparameter Tuning:** Defines, trains, and tunes the RandomForest model.

- **8| XGBoost Classifier with Hyperparameter Tuning:** Defines, trains, and tunes the XGBoost model.

- **9| Model Evaluation and Comparison:** Compares the performance of both models on validation and test sets.

- **10| Detailed Classification Report:** Provides comprehensive classification metrics for both models.

- **11| Confusion Matrix Visualization:** Visualizes the confusion matrices to understand model predictions.

- **12| Feature Importance Analysis:** Analyzes and plots the importance of features for both models.

- **13| Model Performance Summary:** Presents a summary table and plots comparing key performance indicators.



