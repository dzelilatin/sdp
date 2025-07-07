#!/usr/bin/env python3

import matplotlib
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
matplotlib.use('Agg')  # Use non-interactive backend
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import logging
import sqlite3
import warnings
warnings.filterwarnings("ignore")
import json


# Configure logging to write errors to a file
logging.basicConfig(
    filename='error_log.txt',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_results_directory():
    """Create directory for saving results if it doesn't exist."""
    results_dir = 'ml_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir

def add_age_range(df):
    """Add AgeRange column based on SeniorCitizen and tenure."""
    def map_age_range(row):
        if row['SeniorCitizen'] == 'Yes':
            return '66+'
        elif row['tenure'] <= 12:
            return '18-25'
        elif row['tenure'] <= 36:
            return '26-45'
        elif row['tenure'] <= 60:
            return '46-65'
        else:
            return '66+'
    
    df['AgeRange'] = df.apply(map_age_range, axis=1)
    return df


def initialize_database():
    """Initialize SQLite database and create table for predictions."""
    db_path = os.path.join('ml_results', 'predictions.db')

    # Ensure the directory exists
    if not os.path.exists('ml_results'):
        os.makedirs('ml_results')

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_data TEXT,
            prediction INTEGER,
            model_name TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized.")


def save_prediction(input_data, prediction, model_name, column_names):
    """Save prediction and input data with column names to the database."""
    db_path = os.path.join('ml_results', 'predictions.db')
    
    # Convert to JSON with column names
    input_data_json = json.dumps(dict(zip(column_names, input_data)))

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Insert into DB
    cursor.execute('''
        INSERT INTO predictions (input_data, prediction, model_name)
        VALUES (?, ?, ?)
    ''', (input_data_json, prediction, model_name))
    
    conn.commit()
    conn.close()
    print("Prediction saved to database.")

def get_prediction_history():
    """Retrieve all predictions from the database."""
    db_path = os.path.join('ml_results', 'predictions.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Fetch all predictions
    cursor.execute('SELECT * FROM predictions')
    rows = cursor.fetchall()
    conn.close()
    return rows

def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset."""
    print("\n1. Loading and preprocessing data...")
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        print("\nFirst few rows of the dataset:")
        print(df.head())
        
        # Convert SeniorCitizen to categorical for better analysis
        df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
        
        # Create a new TotalCharges column based on MonthlyCharges * tenure
        df['TotalCharges'] = df['MonthlyCharges'] * df['tenure']

        df = add_age_range(df)
        
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        raise

def demographic_analysis(df, results_dir):
    """Perform demographic analysis and save visualizations."""
    print("\n2. Performing demographic analysis...")
    try:
        # Gender Distribution
        plt.figure()
        gender_counts = df['gender'].value_counts()
        plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
        plt.title('Gender Distribution')
        plt.savefig(os.path.join(results_dir, 'gender_distribution.png'))
        plt.close()

        # Senior Citizen Distribution
        plt.figure()
        senior_counts = df['SeniorCitizen'].value_counts()
        plt.pie(senior_counts, labels=senior_counts.index, autopct='%1.1f%%')
        plt.title('Senior Citizen Distribution')
        plt.savefig(os.path.join(results_dir, 'senior_citizen_distribution.png'))
        plt.close()

        print("Demographic analysis completed and visualizations saved.")
    except Exception as e:
        logging.error(f"Error in demographic analysis: {str(e)}")
        raise

def correlation_analysis(df, results_dir):
    """Perform correlation analysis and save heatmap."""
    print("\n3. Performing correlation analysis...")
    try:
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        correlation_matrix = df[numerical_cols].corr()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'correlation_matrix.png'))
        plt.close()

        print("Correlation analysis completed and heatmap saved.")
    except Exception as e:
        logging.error(f"Error in correlation analysis: {str(e)}")
        raise

def service_usage_analysis(df, results_dir):
    """Perform service usage analysis and save visualizations."""
    print("\n4. Performing service usage analysis...")
    try:
        # Internet Service Distribution
        plt.figure()
        internet_counts = df['InternetService'].value_counts()
        plt.pie(internet_counts, labels=internet_counts.index, autopct='%1.1f%%')
        plt.title('Internet Service Distribution')
        plt.savefig(os.path.join(results_dir, 'internet_service_distribution.png'))
        plt.close()

        # Additional Services Distribution
        additional_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                               'TechSupport', 'StreamingTV', 'StreamingMovies']
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        for i, service in enumerate(additional_services):
            row = i // 3
            col = i % 3
            sns.countplot(data=df, x=service, ax=axes[row, col])
            axes[row, col].set_title(f'{service} Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'additional_services_distribution.png'))
        plt.close()

        print("Service usage analysis completed and visualizations saved.")
    except Exception as e:
        logging.error(f"Error in service usage analysis: {str(e)}")
        raise

def payment_analysis(df, results_dir):
    """Perform payment analysis and save visualizations."""
    print("\n5. Performing payment analysis...")
    try:
        # Payment Method Distribution
        plt.figure()
        payment_counts = df['PaymentMethod'].value_counts()
        plt.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%')
        plt.title('Payment Method Distribution')
        plt.savefig(os.path.join(results_dir, 'payment_method_distribution.png'))
        plt.close()

        # Contract Type Distribution
        plt.figure()
        contract_counts = df['Contract'].value_counts()
        plt.pie(contract_counts, labels=contract_counts.index, autopct='%1.1f%%')
        plt.title('Contract Type Distribution')
        plt.savefig(os.path.join(results_dir, 'contract_type_distribution.png'))
        plt.close()

        print("Payment analysis completed and visualizations saved.")
    except Exception as e:
        logging.error(f"Error in payment analysis: {str(e)}")
        raise

def prepare_features(df):
    """Prepare features and target variable."""
    print("\n2. Preparing features and target variable...")
    try:
        # Map Churn to binary (Yes=1, No=0)
        y = df['Churn'].map({'Yes': 1, 'No': 0})
        
        # Columns to drop (only drop if they exist)
        columns_to_drop = ['customerID', 'Churn', 'Race', 'Gender']
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        X = df.drop(existing_columns_to_drop, axis=1)
        
        # One-hot encoding
        X_encoded = pd.get_dummies(X)
        print(f"Number of features after encoding: {X_encoded.shape[1]}")
        
        # Drop low-variance columns
        low_variance_cols = [col for col in X_encoded.columns if X_encoded[col].nunique() == 1]
        X_encoded = X_encoded.drop(columns=low_variance_cols)
        print(f"Number of features after dropping low-variance columns: {X_encoded.shape[1]}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42
        )
        
        # Scaling features ensures all numerical values are on the same scale, improving model performance.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Perform feature selection
        print("\nPerforming feature selection...")
        model = RandomForestClassifier(random_state=42)
        rfe = RFE(model, n_features_to_select=10)  # Select top 10 features
        X_train_selected = rfe.fit_transform(X_train_scaled, y_train)
        X_test_selected = rfe.transform(X_test_scaled)  # Transform test set using fitted selector
        
        print(f"Selected features: {rfe.get_support(indices=True)}")

        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)

        return X_train_resampled, X_test_selected, y_train_resampled, y_test
    
    except Exception as e:
        logging.error(f"Error preparing features: {str(e)}")
        raise

def select_features(X_train, y_train):
    """Perform Recursive Feature Elimination (RFE) for feature selection."""
    print("\nPerforming feature selection...")
    model = RandomForestClassifier(random_state=42)
    rfe = RFE(model, n_features_to_select=10)  # Select top 10 features
    X_train_selected = rfe.fit_transform(X_train, y_train)
    selected_features = rfe.get_support(indices=True)
    print(f"Selected features: {selected_features}")
    return X_train_selected


def train_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple models with detailed metrics."""
    print("\n3. Training and evaluating models...")
    
    # Define parameter grids for each model
    param_grids = {
        'Logistic Regression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],  # Removed 'l1' as it's not supported by 'lbfgs'
            'solver': ['lbfgs']  # Explicitly specify the solver
        },
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.1]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.1, 0.01],
            'max_depth': [3, 5]
        },
        'AdaBoost': {
            'n_estimators': [50, 100],
            'learning_rate': [1.0, 0.5, 1.0]
        },
        'Neural Network': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd']
        },
        'Ridge Classifier': {
            'alpha': [0.1, 1, 10],
            'solver': ['auto', 'svd', 'cholesky']
        },
        'Naive Bayes': {},  # No hyperparameters for GaussianNB
        'KNN': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        },
        'Decision Tree': {
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    }
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Ridge Classifier': RidgeClassifier(),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(probability=True),
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'Neural Network': MLPClassifier(max_iter=2000, random_state=42, solver='adam', learning_rate='adaptive')
    }
    
    # Dictionary to store results
    model_results = {}
    trained_models = {}
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\nTraining {model_name}... This may take some time.")
        start_time = time.time()
        
        try:
            # Perform grid search
            grid_search = GridSearchCV(
                model, 
                param_grids[model_name], 
                cv=5, 
                scoring='recall', 
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            # Get best model and predictions
            best_model = grid_search.best_estimator_
            trained_models[model_name] = best_model 

            predictions = best_model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions)
            recall = recall_score(y_test, predictions)
            false_positives = sum((predictions == 1) & (y_test == 0))
            confusion = confusion_matrix(y_test, predictions)
            report = classification_report(y_test, predictions)
            
            # Store results
            model_results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'false_positives': false_positives,
                'confusion_matrix': confusion,
                'best_params': grid_search.best_params_,
                'report': report
            }
            
            # Print results
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            print(f"Test accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"False Positives: {false_positives}")
            print(f"Training time: {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logging.error(f"Error training {model_name}: {str(e)}")
            print(f"Error training {model_name}. Check error_log.txt for details.")
            continue
    
    return model_results, trained_models

def create_ensemble(X_train, X_test, y_train, y_test, model_results, models):
    """Create and evaluate an ensemble of the top 3 models."""
    print("\n4. Creating ensemble model...")
    try:
        # Get top 3 models
        top_3_models = sorted(
            model_results.items(), 
            key=lambda x: x[1]['accuracy'], 
            reverse=True
        )[:3]
        
        print("\nTop 3 Models:")
        for model_name, results in top_3_models:
            print(f"{model_name}: {results['accuracy']:.4f}")
        
        # Validate estimators
        estimators = [(name, models[name]) for name, _ in top_3_models if name in models]
        if not estimators:
            raise ValueError("Invalid 'estimators' attribute, 'estimators' should be a non-empty list of (string, estimator) tuples.")
        
        # Create ensemble
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        
        # Train ensemble
        start_time = time.time()
        ensemble.fit(X_train, y_train)
        ensemble_pred = ensemble.predict(X_test)
        
        # Evaluate ensemble
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        ensemble_report = classification_report(y_test, ensemble_pred)
        
        print("\nEnsemble Model Performance:")
        print(f"Accuracy: {ensemble_accuracy:.4f}")
        print(f"Training time: {time.time() - start_time:.2f} seconds")
        print("\nClassification Report:")
        print(ensemble_report)
        
        return ensemble_accuracy, ensemble_report
    except Exception as e:
        logging.error(f"Error creating ensemble: {str(e)}")
        raise

def plot_roc_curves(models, X_test, y_test, results_dir):
    """Plot ROC curves for all models and save to the results directory."""
    plt.figure(figsize=(10, 8))
    for model_name, model in models.items():
        try:
            # Check if the model supports predict_proba
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
            else:
                logging.warning(f"{model_name} does not support predict_proba. Skipping ROC curve.")
        except Exception as e:
            logging.error(f"Error plotting ROC for {model_name}: {str(e)}")
            continue
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    
    # Save plot to the results directory
    roc_curve_path = os.path.join(results_dir, 'roc_curves.png')
    plt.savefig(roc_curve_path)
    print(f"ROC curves saved to: {roc_curve_path}")

def plot_results(model_results, results_dir):
    """Generate and save comparison plots for model metrics."""
    print("\n5. Creating performance visualization...")
    try:
        # Extract metrics for comparison
        model_names = list(model_results.keys())
        accuracies = [results['accuracy'] for results in model_results.values()]
        precisions = [results['precision'] for results in model_results.values()]
        recalls = [results['recall'] for results in model_results.values()]
        false_positives = [results['false_positives'] for results in model_results.values()]

        # Create accuracy comparison plot
        plt.figure(figsize=(12, 6))
        plt.bar(model_names, accuracies, color='skyblue')
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy Comparison')
        plt.xticks(rotation=45, ha='right')
        accuracy_plot_path = os.path.join(results_dir, 'model_accuracy_comparison.png')
        plt.tight_layout()
        plt.savefig(accuracy_plot_path)
        print(f"Accuracy comparison plot saved to: {accuracy_plot_path}")
        plt.close()

        # Create model comparison plot (False Positives, Precision, Recall)
        x = range(len(model_names))
        plt.figure(figsize=(12, 6))
        plt.plot(x, false_positives, marker='o', label='False Positives', color='red')
        plt.plot(x, precisions, marker='o', label='Precision', color='green')
        plt.plot(x, recalls, marker='o', label='Recall', color='blue')
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.xlabel('Models')
        plt.ylabel('Metrics')
        plt.title('Model Comparison (False Positives, Precision, Recall)')
        plt.legend()
        comparison_plot_path = os.path.join(results_dir, 'model_metrics_comparison.png')
        plt.tight_layout()
        plt.savefig(comparison_plot_path)
        print(f"Model comparison plot saved to: {comparison_plot_path}")
        plt.close()

    except Exception as e:
        logging.error(f"Error creating performance visualization: {str(e)}")
        raise

def save_results(model_results, ensemble_accuracy, ensemble_report, results_dir):
    """Save all results to a text file with detailed metrics."""
    print("\n6. Saving results...")
    
    try:
        results_path = os.path.join(results_dir, 'model_results.txt')
        with open(results_path, 'w') as f:
            f.write("MODEL PERFORMANCE RESULTS\n")
            f.write("=======================\n\n")
            
            for model_name, results in model_results.items():
                f.write(f"{model_name}:\n")
                f.write(f"Accuracy: {results['accuracy']:.4f}\n")
                f.write(f"Precision: {results['precision']:.4f}\n")
                f.write(f"Recall: {results['recall']:.4f}\n")
                f.write(f"False Positives: {results['false_positives']}\n")
                f.write(f"Confusion Matrix:\n{results['confusion_matrix']}\n")
                f.write(f"Best Parameters: {results['best_params']}\n")
                f.write("\nClassification Report:\n")
                f.write(results['report'])
                f.write("\n" + "="*50 + "\n\n")
            
            f.write("\nENSEMBLE MODEL RESULTS\n")
            f.write("=====================\n")
            f.write(f"Accuracy: {ensemble_accuracy:.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(ensemble_report)
        
        print(f"Results saved to: {results_path}")
        
    except Exception as e:
        logging.error(f"Error saving results: {str(e)}")
        print(f"Error saving results. Check error_log.txt for details.")
        raise


def main():
    """Main function to run the analysis."""
    try:
        # Create results directory
        results_dir = create_results_directory()
        
        # Load and preprocess data
        data_path = '/Users/user/Downloads/telcodataset.csv'
        df = load_and_preprocess_data(data_path)

        demographic_analysis(df, results_dir)
        correlation_analysis(df, results_dir)
        service_usage_analysis(df, results_dir)
        payment_analysis(df, results_dir)
        
        # Prepare features
        X_train, X_test, y_train, y_test = prepare_features(df)
        
        # Train models
        model_results, trained_models = train_models(X_train, X_test, y_train, y_test)
        
        # Initialize models (this was missing in the original code)
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Ridge Classifier': RidgeClassifier(),
            'Naive Bayes': GaussianNB(),
            'SVM': SVC(probability=True),
            'KNN': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'Neural Network': MLPClassifier(max_iter=3000, random_state=42, solver='adam', learning_rate='adaptive')
        }
        
        # Create ensemble
        ensemble_accuracy, ensemble_report = create_ensemble(
            X_train, X_test, y_train, y_test, model_results, models
        )

        # Plot ROC curves
        plot_roc_curves(trained_models, X_test, y_test, results_dir)
        
        # Create visualization
        plot_results(model_results, results_dir)
        
        # Save results
        save_results(model_results, ensemble_accuracy, ensemble_report, results_dir)
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        print(f"Error in main execution. Check error_log.txt for details.")
        raise

if __name__ == "__main__":
    main()