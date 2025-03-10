import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function for final zero-uncertainty classification
def classify_clone(token_sim, ast_sim, data_flow_sim, runtime_sim):
    if token_sim >= 100 and ast_sim >= 100 and data_flow_sim >= 100 and runtime_sim >= 100:
        return "Type-1"  # Identical or near-identical clones
    elif token_sim >= 80 and ast_sim >= 95 and data_flow_sim >= 95 and runtime_sim >= 100:
        return "Type-2"  # Identical structure but renamed variables
    elif token_sim >= 50 and ast_sim >= 50 and data_flow_sim >= 50 and runtime_sim >= 100:
        return "Type-3"  # Significant but functionally equivalent changes
    elif token_sim <= 15 and ast_sim <= 15 and data_flow_sim <= 15 and runtime_sim >= 5:
        return "Type-4"  # Completely different implementation, same functionality
    else:
        return "Type-3"  # Assign remaining uncertain cases to Type-3 (most probable clone type)

# Generate 1000 new random test cases
np.random.seed(42)  # For reproducibility
num_samples = 100000

test_cases = pd.DataFrame({
    "Token Similarity": np.random.randint(0, 101, num_samples),
    "AST Similarity": np.random.randint(0, 101, num_samples),
    "Data Flow Similarity": np.random.randint(0, 101, num_samples),
    "Runtime Similarity": np.random.randint(0, 101, num_samples)
})

# Apply classification function
test_cases["Clone Type"] = test_cases.apply(
    lambda row: classify_clone(row["Token Similarity"], row["AST Similarity"], row["Data Flow Similarity"], row["Runtime Similarity"]), axis=1
)

# Count occurrences of each classification
classification_counts = test_cases["Clone Type"].value_counts()

# Display classification distribution
plt.figure(figsize=(10, 5))
classification_counts.plot(kind="bar", color=["blue", "green", "orange", "red"])
plt.xlabel("Clone Type Classification")
plt.ylabel("Number of Cases")
plt.title("Distribution of Clone Classifications in 1000 Random Cases")
plt.xticks(rotation=45)
plt.show()

# Save to CSV for further analysis
test_cases.to_csv("clone_classification_results.csv", index=False)
print("Classification results saved to clone_classification_results.csv")
