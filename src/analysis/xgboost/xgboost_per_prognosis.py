# %%
from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# %% Find data folder
def find_directory(target_dir):
    return next((p / target_dir for p in Path.cwd().parents if (p / target_dir).is_dir()), None) or Path(target_dir)

# Define the path to the data folder
data_dir = find_directory('data')
print(f'{data_dir=}')

# %%
# Load the data
raw_data_dir = data_dir / 'raw'
train_data = pd.read_csv(raw_data_dir / 'Training.csv') 
test_data = pd.read_csv(raw_data_dir / 'Testing.csv')

# %%
# Drop unnamed last column containing all NaNs
train_data.dropna(how='all', axis='columns', inplace=True)
assert train_data.shape == (4920,133), "Training data should have 4920 rows and 133 cols after all NaNs col is removed"
print(train_data.columns)

# %%
# Separate features and labels
label_col = "prognosis"
X_train = train_data.drop(label_col, axis=1)
y_train = train_data[label_col]
X_test = test_data.drop(label_col, axis=1)
y_test = test_data[label_col]

# %%
# Encode prognoses
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# %%
# Train a model for each prognosis and evaluate
prognoses = le.classes_
n_prognoses = len(prognoses)
models = []
evaluation_results = []

for i in range(n_prognoses):
    prognosis_name = prognoses[i]
    print(f"Training and evaluating model for: {prognosis_name}")
    
    # Create binary target: 1 for current prognosis, 0 for others
    y_train_binary = (y_train_encoded == i).astype(int)
    y_test_binary = (y_test_encoded == i).astype(int)
    
    # Train XGBoost model
    model = XGBClassifier(eval_metric='logloss')
    model.fit(X_train, y_train_binary)
    models.append(model)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate model
    accuracy = accuracy_score(y_test_binary, y_pred)
    precision = precision_score(y_test_binary, y_pred, zero_division=0)
    recall = recall_score(y_test_binary, y_pred, zero_division=0)
    f1 = f1_score(y_test_binary, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_test_binary, y_pred_proba)
    
    evaluation_results.append({
        'Prognosis': prognosis_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc_roc
    })

# %%
# Convert evaluation results to DataFrame
eval_df = pd.DataFrame(evaluation_results)
print(eval_df)

# %%
# Function to plot top N important features for a prognosis
def plot_top_features(model, prognosis_name, n=10):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_features = X_train.columns[indices[:n]]
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Top {n} Important Symptoms for {prognosis_name}")
    plt.bar(range(n), importances[indices[:n]])
    plt.xticks(range(n), top_features, rotation=90)
    plt.tight_layout()
    plt.show()

# Plot top features for each prognosis
for prognosis, model in zip(prognoses, models):
    plot_top_features(model, prognosis)

# %%
# Save evaluation results and feature importances
output_dir = data_dir / 'analysis' / 'xgboost'
eval_df.to_csv(output_dir / 'prognosis_model_evaluations.csv', index=False)
all_importances = pd.DataFrame(
    {prognosis: model.feature_importances_ for prognosis, model in zip(prognoses, models)},
    index=X_train.columns
)
all_importances.to_csv(output_dir / 'symptom_importances_by_prognosis.csv')

# %%
# Plot sum of feature importances across all models (features on axis-x)

# Sum up feature importances across all models
feature_importance_sum = pd.DataFrame(0, index=X_train.columns, columns=['Importance'])

for model in models:
    importance = model.feature_importances_
    feature_importance_sum['Importance'] += importance

# Sort the summed importances in descending order
feature_importance_sum = feature_importance_sum.sort_values('Importance', ascending=False)

# Create a bar plot
plt.figure(figsize=(12, 8))
plt.bar(feature_importance_sum.index, feature_importance_sum['Importance'])
plt.title('Sum of Feature Importances Across All One-vs-All Models')
plt.xlabel('Features')
plt.ylabel('Summed Importance')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Optionally, save the plot
plt.savefig(output_dir / 'feature_importance_sum.png')

# Print the top N important features
N = 10  # Change this to show more or fewer features
print(f"Top {N} important features:")
print(feature_importance_sum.head(N))

# %%
# Plot sum of feature importances across all models (features on axis-y)

# Sort the summed importances in descending order
feature_importance_sum = feature_importance_sum.sort_values('Importance', ascending=True)

# Create a horizontal bar plot
plt.figure(figsize=(12, 20))  # Adjust the figure size as needed
plt.barh(feature_importance_sum.index, feature_importance_sum['Importance'])
plt.title('Sum of Feature Importances Across All One-vs-All Models')
plt.xlabel('Summed Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# Optionally, save the plot
plt.savefig(output_dir / 'feature_importance_sum_horizontal.png', bbox_inches='tight')

# Print the top N important features
N = 10  # Change this to show more or fewer features
print(f"Top {N} important features:")
print(feature_importance_sum.tail(N)[::-1])  # Reverse to show highest importance first

