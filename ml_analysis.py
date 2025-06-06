#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

def create_results_directory():
    """Create directory for saving results if it doesn't exist."""
    results_dir = 'ml_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir

def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset."""
    print("\n1. Loading and preprocessing data...")
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        print("\nFirst few rows of the dataset:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise

def prepare_features(df):
    """Prepare features and target variable."""
    print("\n2. Preparing features and target variable...")
    try:
        # Map Churn to binary (Yes=1, No=0)
        y = df['Churn'].map({'Yes': 1, 'No': 0})
        X = df.drop(['customerID', 'Churn'], axis=1)
        
        # One-hot encoding
        X_encoded = pd.get_dummies(X)
        print(f"Number of features after encoding: {X_encoded.shape[1]}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    except Exception as e:
        print(f"Error preparing features: {str(e)}")
        raise

def train_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple models."""
    print("\n3. Training and evaluating models...")
    
    # Define parameter grids for each model
    param_grids = {
        'Logistic Regression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2']
        },
        'Random Forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10]
        },
        'XGBoost': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
    }
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'SVM': SVC(probability=True)
    }
    
    # Dictionary to store results
    model_results = {}
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        start_time = time.time()
        
        try:
            # Perform grid search
            grid_search = GridSearchCV(
                model, 
                param_grids[model_name], 
                cv=5, 
                scoring='accuracy', 
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            # Get best model and predictions
            best_model = grid_search.best_estimator_
            predictions = best_model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions)
            
            # Store results
            model_results[model_name] = {
                'accuracy': accuracy,
                'predictions': predictions,
                'best_params': grid_search.best_params_,
                'report': report
            }
            
            # Print results
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            print(f"Test accuracy: {accuracy:.4f}")
            print(f"Training time: {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            continue
    
    return model_results

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
        
        # Create ensemble
        estimators = [(name, models[name]) for name, _ in top_3_models]
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
        print(f"Error creating ensemble: {str(e)}")
        raise

def plot_results(model_results, results_dir):
    """Create and save visualization of model performances."""
    print("\n5. Creating performance visualization...")
    
    try:
        plt.figure(figsize=(12, 6))
        accuracies = [results['accuracy'] for results in model_results.values()]
        model_names = list(model_results.keys())
        
        plt.bar(model_names, accuracies)
        plt.xticks(rotation=45)
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(results_dir, 'model_comparison.png')
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")
        
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        raise

def save_results(model_results, ensemble_accuracy, ensemble_report, results_dir):
    """Save all results to a text file."""
    print("\n6. Saving results...")
    
    try:
        results_path = os.path.join(results_dir, 'model_results.txt')
        with open(results_path, 'w') as f:
            f.write("MODEL PERFORMANCE RESULTS\n")
            f.write("=======================\n\n")
            
            for model_name, results in model_results.items():
                f.write(f"{model_name}:\n")
                f.write(f"Accuracy: {results['accuracy']:.4f}\n")
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
        print(f"Error saving results: {str(e)}")
        raise

def main():
    """Main function to run the analysis."""
    try:
        # Create results directory
        results_dir = create_results_directory()
        
        # Load and preprocess data
        data_path = '/Users/user/Downloads/telcodataset.csv'
        df = load_and_preprocess_data(data_path)
        
        # Prepare features
        X_train, X_test, y_train, y_test = prepare_features(df)
        
        # Train models
        model_results = train_models(X_train, X_test, y_train, y_test)
        
        # Create ensemble
        ensemble_accuracy, ensemble_report = create_ensemble(
            X_train, X_test, y_train, y_test, model_results, models
        )
        
        # Create visualization
        plot_results(model_results, results_dir)
        
        # Save results
        save_results(model_results, ensemble_accuracy, ensemble_report, results_dir)
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 