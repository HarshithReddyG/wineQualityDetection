import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_and_save(input_path="../data/winequality-white.csv", 
                        output_dataset="../data/winequality-white-cleaned.csv", 
                        output_log="../output/dataPreprocessing/dataPreprocessing.txt"):
    
    # Create necessary directories
    os.makedirs("../output/dataPreprocessing/graphs", exist_ok=True)
    os.makedirs("../output/dataPreprocessing", exist_ok=True)

    # Open log file for writing
    log_file = open(output_log, "w")

    # Load dataset
    dataset = pd.read_csv(input_path, delimiter=';')
    log_file.write(f"Initial dataset shape: {dataset.shape}\n")
    
    # Check for missing values
    missing_values = dataset.isnull().sum()
    log_file.write(f"Missing values in each column:\n{missing_values}\n")
    
    # Check for duplicate rows
    duplicates = dataset.duplicated().sum()
    log_file.write(f"Number of duplicate rows: {duplicates}\n")
    dataset = dataset.drop_duplicates()
    log_file.write(f"Dataset shape after removing duplicates: {dataset.shape}\n")
    
    # Handle outliers using z-score
    z_scores = dataset.iloc[:, :-1].apply(zscore)
    threshold = 3
    outliers = (z_scores.abs() > threshold).any(axis=1)
    outliers_count = outliers.sum()
    dataset = dataset[~outliers]
    log_file.write(f"Number of outliers removed: {outliers_count}\n")
    log_file.write(f"Dataset shape after removing outliers: {dataset.shape}\n")
    
    # Apply log transformation for skewed features
    skewed_features = ['residual sugar', 'chlorides']
    for feature in skewed_features:
        dataset[feature] = np.log1p(dataset[feature])  # log1p avoids log(0) issues
    log_file.write(f"Log transformation applied to features: {skewed_features}\n")
    
    # Drop features with negligible correlation
    dataset = dataset.drop(['free sulfur dioxide', 'citric acid'], axis=1)
    log_file.write("Dropped features: free sulfur dioxide, citric acid\n")
    
    # Categorize quality into low, medium, and high
    def categorize_quality(quality):
        if quality <= 3:
            return 'low'
        elif quality <= 6:
            return 'medium'
        else:
            return 'high'
    
    dataset['quality_category'] = dataset['quality'].apply(categorize_quality)
    dataset = dataset.drop('quality', axis=1)
    
    # Separate features and target
    X = dataset.drop('quality_category', axis=1)
    y = dataset['quality_category']
    
    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    log_file.write("Applied SMOTE for class imbalance.\n")
    log_file.write(f"Class distribution after resampling:\n{pd.Series(y_resampled).value_counts()}\n")
    
    # Encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_resampled)
    log_file.write(f"Encoded target variable with mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}\n")
    
    # Cap extreme outliers for chlorides
    z_scores_chlorides = zscore(dataset['chlorides'])
    outlier_mask = z_scores_chlorides.abs() > threshold
    dataset.loc[outlier_mask, 'chlorides'] = dataset['chlorides'].quantile(0.99)
    log_file.write("Capped extreme outliers in 'chlorides'.\n")
    
    # Apply feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    log_file.write("Feature scaling applied to resampled data.\n")
    
    # Save cleaned dataset
    dataset['quality_category'] = label_encoder.transform(dataset['quality_category'])
    dataset.to_csv(output_dataset, index=False)
    log_file.write(f"Cleaned dataset saved at {output_dataset}\n")
    
    # Visualization
    # Boxplot for feature distributions
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=dataset.iloc[:, :-1])
    plt.title("Feature Distributions with Outliers")
    plt.xticks(rotation=45)
    plt.savefig("../output/dataPreprocessing/graphs/boxplot_features.png")
    plt.close()
    
    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.savefig("../output/dataPreprocessing/graphs/correlation_heatmap.png")
    plt.close()
    
    # Class distribution
    plt.figure(figsize=(8, 6))
    dataset['quality_category'] = dataset['quality_category'].replace({0: 'low', 1: 'medium', 2: 'high'})
    dataset['quality_category'].value_counts().plot(kind='bar', color=['blue', 'green', 'red'])
    plt.title("Class Distribution")
    plt.xlabel("Quality Category")
    plt.ylabel("Frequency")
    plt.savefig("../output/dataPreprocessing/graphs/class_distribution.png")
    plt.close()
    
    log_file.close()
    print(f"Preprocessing completed. Logs saved at {output_log}")

    # Return the preprocessed dataset
    return pd.DataFrame(X_scaled, columns=X.columns), y_encoded


