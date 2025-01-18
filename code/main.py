from dataPreprocessing import preprocess_and_save
from modelbuilding import build_models
from visualisation import evaluate_and_save_results

# Preprocess the dataset and get the cleaned dataset
X, y = preprocess_and_save()

# Build models and get predictions
results = build_models(X, y)

# Evaluate and visualize results
evaluate_and_save_results(results, output_dir="../output")
