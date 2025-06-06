import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create results directory if it doesn't exist
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_results')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Load and preprocess data
print("Loading and preprocessing data...")
df = pd.read_csv('/Users/user/Downloads/telcodataset.csv')

# Separate target before encoding
# Map Churn to binary if not already (Yes=1, No=0)
y = df['Churn'].map({'Yes': 1, 'No': 0})
X = df.drop(['customerID', 'Churn'], axis=1)
X_encoded = pd.get_dummies(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dictionary to store model results
model_results = {}

# 1. Logistic Regression
print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
model_results['Logistic Regression'] = {
    'accuracy': accuracy_score(y_test, lr_pred),
    'predictions': lr_pred
}

# 2. Ridge Classifier
print("Training Ridge Classifier...")
ridge = RidgeClassifier()
ridge.fit(X_train_scaled, y_train)
ridge_pred = ridge.predict(X_test_scaled)
model_results['Ridge Classifier'] = {
    'accuracy': accuracy_score(y_test, ridge_pred),
    'predictions': ridge_pred
}

# 3. Naive Bayes
print("Training Naive Bayes...")
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
nb_pred = nb.predict(X_test_scaled)
model_results['Naive Bayes'] = {
    'accuracy': accuracy_score(y_test, nb_pred),
    'predictions': nb_pred
}

# 4. SVM
print("Training SVM...")
svm = SVC(probability=True)
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)
model_results['SVM'] = {
    'accuracy': accuracy_score(y_test, svm_pred),
    'predictions': svm_pred
}

# 5. KNN
print("Training KNN...")
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)
knn_pred = knn.predict(X_test_scaled)
model_results['KNN'] = {
    'accuracy': accuracy_score(y_test, knn_pred),
    'predictions': knn_pred
}

# 6. Decision Tree
print("Training Decision Tree...")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)
dt_pred = dt.predict(X_test_scaled)
model_results['Decision Tree'] = {
    'accuracy': accuracy_score(y_test, dt_pred),
    'predictions': dt_pred
}

# 7. Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)
model_results['Random Forest'] = {
    'accuracy': accuracy_score(y_test, rf_pred),
    'predictions': rf_pred
}

# 8. XGBoost
print("Training XGBoost...")
xgb = XGBClassifier(random_state=42)
xgb.fit(X_train_scaled, y_train)
xgb_pred = xgb.predict(X_test_scaled)
model_results['XGBoost'] = {
    'accuracy': accuracy_score(y_test, xgb_pred),
    'predictions': xgb_pred
}

# Print results
print("\nModel Performance Comparison:")
for model_name, results in model_results.items():
    print(f"\n{model_name}:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, results['predictions']))

# Create accuracy comparison plot
plt.figure(figsize=(12, 6))
accuracies = [results['accuracy'] for results in model_results.values()]
model_names = list(model_results.keys())
plt.bar(model_names, accuracies)
plt.xticks(rotation=45)
plt.title('Model Accuracy Comparison')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'model_accuracy_comparison.png'))
plt.close()

# Find top 3 models
top_3_models = sorted(model_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:3]
print("\nTop 3 Models:")
for model_name, results in top_3_models:
    print(f"{model_name}: {results['accuracy']:.4f}")

# Create ensemble with top 3 models
print("\nCreating ensemble with top 3 models...")
estimators = []
for model_name, _ in top_3_models:
    if model_name == 'Logistic Regression':
        estimators.append(('lr', lr))
    elif model_name == 'Ridge Classifier':
        estimators.append(('ridge', ridge))
    elif model_name == 'Naive Bayes':
        estimators.append(('nb', nb))
    elif model_name == 'SVM':
        estimators.append(('svm', svm))
    elif model_name == 'KNN':
        estimators.append(('knn', knn))
    elif model_name == 'Decision Tree':
        estimators.append(('dt', dt))
    elif model_name == 'Random Forest':
        estimators.append(('rf', rf))
    elif model_name == 'XGBoost':
        estimators.append(('xgb', xgb))

ensemble = VotingClassifier(estimators=estimators, voting='soft')
ensemble.fit(X_train_scaled, y_train)
ensemble_pred = ensemble.predict(X_test_scaled)

print("\nEnsemble Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, ensemble_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, ensemble_pred))

# Save results to file
with open(os.path.join(results_dir, 'model_results.txt'), 'w') as f:
    f.write("Model Performance Comparison:\n")
    for model_name, results in model_results.items():
        f.write(f"\n{model_name}:\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, results['predictions']))
    
    f.write("\nEnsemble Model Performance:\n")
    f.write(f"Accuracy: {accuracy_score(y_test, ensemble_pred):.4f}\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_test, ensemble_pred)) 