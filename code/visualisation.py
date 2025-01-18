import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_and_save_results(results, output_dir="../output"):
    """Evaluate and save results for all models."""
    # Create the main output directory
    if not isinstance(output_dir, str):
        raise ValueError("Expected a valid directory path as 'output_dir'.")
    os.makedirs(output_dir, exist_ok=True)

    all_results_file = os.path.join(output_dir, "model_results.txt")
    best_accuracy_file = os.path.join(output_dir, "bestAccuracy.txt")
    graphs_dir = os.path.join(output_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    best_model = None
    best_accuracy = 0
    best_results = None

    # Write results and save graphs
    with open(all_results_file, "w") as file:
        for model_name, y_pred in results['predictions'].items():
            y_test = results['y_test']
            accuracy = results['accuracies'][model_name]
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred)

            # Write to results file
            file.write(f"\n{model_name} Results:\n")
            file.write(f"Accuracy: {accuracy * 100:.2f}%\n")
            file.write("Confusion Matrix:\n")
            file.write(f"{conf_matrix}\n")
            file.write("Classification Report:\n")
            file.write(f"{class_report}\n")

            # Save confusion matrix heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues',
                        xticklabels=['low', 'medium', 'high'], yticklabels=['low', 'medium', 'high'])
            plt.title(f"Confusion Matrix for {model_name}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.savefig(os.path.join(graphs_dir, f"{model_name}_confusion_matrix.png"))
            plt.close()

            # Update best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_name
                best_results = {
                    "accuracy": accuracy,
                    "conf_matrix": conf_matrix,
                    "class_report": class_report
                }

    # Save best model results
    if best_results:
        with open(best_accuracy_file, "w") as best_file:
            best_file.write(f"Best Model: {best_model}\n")
            best_file.write(f"Accuracy: {best_results['accuracy'] * 100:.2f}%\n")
            best_file.write("Confusion Matrix:\n")
            best_file.write(f"{best_results['conf_matrix']}\n")
            best_file.write("Classification Report:\n")
            best_file.write(f"{best_results['class_report']}\n")

    # Save accuracy comparison chart directly in ../output
    plt.figure(figsize=(10, 6))
    plt.bar(results['accuracies'].keys(), [v * 100 for v in results['accuracies'].values()], color='skyblue')
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=45)
    comparison_chart_path = os.path.join(output_dir, "model_accuracy_comparison.png")
    plt.savefig(comparison_chart_path)
    plt.close()
    print(f"Model accuracy comparison saved at {comparison_chart_path}")
