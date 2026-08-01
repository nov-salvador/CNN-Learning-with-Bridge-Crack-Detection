import csv

from configs.config import *

import matplotlib.pyplot as plt

class TestLogger():
  def __init__(self):
    TEST_OUTPUT.mkdir(parents=True, exist_ok=True)
    if not TEST_CSV_FILE.exists():
      with open(TEST_CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
          "Model Name",
          "Accuracy", 
          "Precision",
          "Recall",   
          "F1 Score", 
          "Loss",
        ])

  def log(self, metrics):
    with open(TEST_CSV_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
          MODEL_NAME, 
          f"{metrics["accuracy"]:.4f}", 
          f"{metrics["precision"]:.4f}",
          f"{metrics["recall"]:.4f}",
          f"{metrics["f1"]:.4f}",
          f"{metrics["loss"]:.4f}",
          ])

  def log_classification_report(self, report):
    with open(TEST_CLASSIFICATION_REPORT, "w") as f:
      f.write(report)

  def log_conf_matrix(self, matrix):
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar()

    plt.xticks([0, 1], ["crack", "no_crack"])
    plt.yticks([0, 1], ["crack", "no_crack"])

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(2):
      for j in range(2):
        plt.text(j, i, matrix[i, j],
                 ha="center",
                 va="center")

    plt.title("Confusion Matrix")
    
    plt.tight_layout()

    plt.savefig(TEST_CONF_MATRIX)

    plt.close()
    

    
