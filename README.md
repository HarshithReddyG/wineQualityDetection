# Wine Quality Prediction Using Machine Learning

## Overview
This project predicts wine quality as **low**, **medium**, or **high** using a stacking model that combines **Random Forest** and **LightGBM**. The project workflow includes data preprocessing, model building, and visualizing results in a **Streamlit** application.

## Workflow
### 1. Data Preprocessing
- **File:** `dataPreprocessing.py`
- **Key Steps:**
  1. Load and clean the dataset by removing duplicates and handling missing values.
  2. Handle class imbalance using **SMOTE**.
  3. Apply log transformations for skewed features.
  4. Scale features using **StandardScaler**.
  5. Save the scaler and feature names for consistent input handling.
  6. Save the cleaned dataset as `winequality-white-cleaned.csv`.

- **Outputs:**
  - `../data/winequality-white-cleaned.csv` (cleaned dataset)
  - `../models/scaler.joblib` (scaler for feature scaling)
  - `../models/feature_names.joblib` (feature names for consistency)

### 2. Model Building
- **File:** `modelbuilding.py`
- **Key Steps:**
  1. Train individual models: Logistic Regression, Random Forest, LightGBM.
  2. Create a **Stacking Classifier** combining Random Forest and LightGBM, with Logistic Regression as the final estimator.
  3. Evaluate all models and save the best one based on accuracy.
  4. Save the stacking model as `stacking_model.joblib`.

- **Outputs:**
  - `../models/stacking_model.joblib` (saved stacking model)

### 3. Streamlit App
- **File:** `app.py`
- **Key Features:**
  1. Accepts user input for wine features through sliders.
  2. Loads the saved scaler, feature names, and stacking model.
  3. Scales user input and predicts wine quality.
  4. Displays prediction probabilities with a bar chart.

- **Commands:**
  ```bash
  streamlit run app.py
  ```

### 4. Visualizations
- Feature distributions are visualized with boxplots.
- Model comparison results are displayed with accuracy charts.
- Confusion matrices for models are plotted and saved.

## File Structure
```
WineQualityDetection/
├── data/
│   ├── winequality-white.csv
│   └── winequality-white-cleaned.csv
├── models/
│   ├── scaler.joblib
│   ├── feature_names.joblib
│   └── stacking_model.joblib
├── output/
│   ├── dataPreprocessing/
│   │   ├── logs.txt
│   │   └── graphs/
│   ├── logisticRegression/
│   │   ├── results.txt
│   │   └── graphs/
│   ├── bestModelResults/
│   │   ├── results.txt
│   │   └── graphs/
│   └── model_accuracy_comparison.png
├── code/
│   ├── dataPreprocessing.py
│   ├── modelbuilding.py
│   └── app.py
└── README.md
```

## Key Features
1. **Consistent Feature Handling:** Ensures feature names match between preprocessing and the app.
2. **Stacking Model:** Combines the strengths of Random Forest and LightGBM.
3. **Streamlit Visualization:** Interactive web app for predictions and results.
4. **Modular Design:** Separate files for preprocessing, model building, and app logic.

## How to Run
1. **Preprocess the Data:**
   ```bash
   python dataPreprocessing.py
   ```
2. **Build Models:**
   ```bash
   python modelbuilding.py
   ```
3. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

## Requirements
- Python 3.9+
- Required Libraries:
  - pandas
  - numpy
  - scikit-learn
  - imbalanced-learn
  - joblib
  - streamlit
  - lightgbm

Install dependencies:
```bash
pip install -r requirements.txt
```

