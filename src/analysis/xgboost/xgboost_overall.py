# %%
from pathlib import Path
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# %% Check XGBoost version
def versiontuple(v):
    return tuple(map(int, (v.split("."))))
print(f'{xgb.__version__=}')
assert versiontuple(xgb.__version__) >= versiontuple("2.1.1")

# %% Find data folder
def find_directory(target_dir):
    return next((p / target_dir for p in Path.cwd().parents if (p / target_dir).is_dir()), None) or Path(target_dir)

# Define the path to the data folder
data_dir = find_directory('data') or Path('data')
print(f'{data_dir=}')

# %% Load the data
raw_data_dir = data_dir / 'raw'
train_data = pd.read_csv(raw_data_dir / 'Training.csv') 
test_data = pd.read_csv(raw_data_dir / 'Testing.csv')

# %% Drop unnamed last column containing all NaNs
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
# Encode the disease labels into numerical format
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# %%
# Define the XGBoost classifier with multi-label strategy
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(label_encoder.classes_))

# Train the model
xgb_model.fit(X_train, y_train_encoded)

# Make predictions
y_pred = xgb_model.predict(X_test)

# %%
# Evaluate the model
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Print classification report for detailed performance metrics
print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))

# %%
# Feature importance analysis
feature_importances = xgb_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("Top 10 Important Features Overall:")
print(feature_importance_df.head(10))
print()
print("Top 10 Unimportant Features Overall:")
print(feature_importance_df.tail(10))

# %%
# Plot Features per importance
plt.figure(figsize=(20,20))
sns.barplot(x='Importance', y= 'Feature', data=feature_importance_df)
plt.title('All Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.grid()
plt.show()
# %%
