#!/usr/bin/env python3

import matplotlib
from sklearn.feature_selection import SelectKBest, chi2
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
import argparse
import joblib
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
    df['AgeRange'] = df['SeniorCitizen'].apply(lambda x: '66+' if x == 'Yes' else '18-65')
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

def save_models(trained_models, path='ml_results/models/'):
    os.makedirs(path, exist_ok=True)
    for name, model in trained_models.items():
        joblib.dump(model, os.path.join(path, f"{name.replace(' ', '_')}.joblib"))
    print("Models saved.")

def load_models(path='ml_results/models/'):
    models = {}
    for file in os.listdir(path):
        if file.endswith('.joblib'):
            model_name = file.replace('_', ' ').replace('.joblib', '')
            models[model_name] = joblib.load(os.path.join(path, file))
    return models

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


def demographic_analysis(df, vis_dir):
    """Perform demographic analysis and save visualizations."""
    print("\nPerforming demographic analysis...")
    try:
        custom_colors = ['#4169E1', '#E74C3C']  # Royal Blue & Red for binary pies

        # Gender Distribution
        plt.figure()
        gender_counts = df['gender'].value_counts()
        plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=custom_colors)
        plt.title('Gender Distribution')
        plt.savefig(os.path.join(vis_dir, 'gender_distribution.png'))
        plt.close()

        # Senior Citizen Distribution
        plt.figure()
        senior_counts = df['SeniorCitizen'].value_counts()
        plt.pie(senior_counts, labels=senior_counts.index, autopct='%1.1f%%', colors=custom_colors)
        plt.title('Senior Citizen Distribution')
        plt.savefig(os.path.join(vis_dir, 'seniorcitizen_distribution.png'))
        plt.close()

        # Partner Distribution
        plt.figure()
        counts = df['Partner'].value_counts()
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=custom_colors)
        plt.title('Partner Distribution')
        plt.savefig(os.path.join(vis_dir, 'partner_distribution.png'))
        plt.close()

        # Dependents Distribution
        plt.figure()
        counts = df['Dependents'].value_counts()
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=custom_colors)
        plt.title('Dependents Distribution')
        plt.savefig(os.path.join(vis_dir, 'dependents_distribution.png'))
        plt.close()

        print("Demographic analysis completed and visualizations saved.")
    except Exception as e:
        logging.error(f"Error in demographic analysis: {str(e)}")
        raise


def correlation_analysis(df, vis_dir):
    """Perform correlation analysis and save heatmap."""
    print("\nPerforming correlation analysis...")
    try:
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        correlation_matrix = df[numerical_cols].corr()

        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'correlation_matrix.png'))
        plt.close()

        print("Correlation analysis completed and heatmap saved.")
    except Exception as e:
        logging.error(f"Error in correlation analysis: {str(e)}")
        raise


def internet_service_analysis(df, vis_dir):
    """Visualizations related to internet service."""
    print("\nPerforming internet service analysis...")
    try:
        colors = ['#4169E1', '#E74C3C', '#FF69B4']  # Royal Blue, Red, Pink

        # Internet Service Distribution
        plt.figure()
        counts = df['InternetService'].value_counts()
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=colors)
        plt.title('Internet Service Distribution')
        plt.savefig(os.path.join(vis_dir, 'internetservice_distribution.png'))
        plt.close()

        # Internet Service by Senior Citizen (barplot)
        plt.figure(figsize=(8,6))
        sns.countplot(x='InternetService', hue='SeniorCitizen', data=df, palette=['#4169E1', '#E74C3C'])
        plt.title('Internet Service by Senior Citizen')
        plt.savefig(os.path.join(vis_dir, 'internetservice_senior.png'))
        plt.close()

        # Internet Service vs Churn
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x='InternetService', hue='Churn', palette=colors)
        plt.title('Internet Service vs Churn')
        plt.savefig(os.path.join(vis_dir, 'internetservice_churn.png'))
        plt.close()

        print("Internet service visualizations saved.")
    except Exception as e:
        logging.error(f"Error in internet service analysis: {str(e)}")
        raise

def additional_services_churn(df, vis_dir):
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    colors = ['#4169E1', '#E74C3C']  # Blue & Red for churn categories

    for service in services:
        plt.figure()
        sns.countplot(data=df, x=service, hue='Churn', palette=colors)
        plt.title(f'{service} vs Churn')
        plt.savefig(os.path.join(vis_dir, f'{service.lower()}_churn.png'))
        plt.close()

def internet_service_senior(df, vis_dir):
    plt.figure(figsize=(8,6))
    sns.countplot(x='InternetService', hue='SeniorCitizen', data=df, palette=['#4169E1', '#E74C3C'])
    plt.title('Internet Service by Senior Citizen')
    plt.savefig(os.path.join(vis_dir, 'internet_service_senior.png'))
    plt.close()

def service_usage_analysis(df, vis_dir):
    """Visualizations for additional services and their relation to churn."""
    print("\nPerforming service usage analysis...")
    try:
        services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        palette = ['#4169E1', '#E74C3C']  # Yes/No colors
        
        # Additional Services Distribution (all in one figure)
        fig, axes = plt.subplots(2, 3, figsize=(18,12))
        for i, service in enumerate(services):
            ax = axes[i//3, i%3]
            sns.countplot(x=service, data=df, palette=palette, ax=ax)
            ax.set_title(f'{service} Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'additional_services_distribution.png'))
        plt.close()

        # Each Service vs Churn
        for service in services:
            plt.figure()
            sns.countplot(x=service, hue='Churn', data=df, palette=palette)
            plt.title(f'{service} vs Churn')
            plt.savefig(os.path.join(vis_dir, f'{service.lower()}_churn.png'))
            plt.close()

        print("Service usage visualizations saved.")
    except Exception as e:
        logging.error(f"Error in service usage analysis: {str(e)}")
        raise

def online_backup_distribution(df, vis_dir):
    plt.figure()
    sns.countplot(x='OnlineBackup', data=df, palette=['#4169E1', '#E74C3C'])
    plt.title('Online Backup Distribution')
    plt.savefig(os.path.join(vis_dir, 'OnlineBackup_distribution.png'))
    plt.close()

def online_security_distribution(df, vis_dir):
    plt.figure()
    sns.countplot(x='OnlineSecurity', data=df, palette=['#4169E1', '#E74C3C'])
    plt.title('Online Security Distribution')
    plt.savefig(os.path.join(vis_dir, 'OnlineSecurity_distribution.png'))
    plt.close()


def payment_and_contract_analysis(df, vis_dir):
    """Visualizations for payment methods and contract types."""
    print("\nPerforming payment and contract analysis...")
    try:
        colors_4 = ['#4169E1', '#E74C3C', '#FF69B4', '#2ECC71']  # Blue, Red, Pink, Green
        colors_3 = ['#4169E1', '#E74C3C', '#FF69B4']

        # Payment Method Distribution
        plt.figure()
        counts = df['PaymentMethod'].value_counts()
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=colors_4)
        plt.title('Payment Method Distribution')
        plt.savefig(os.path.join(vis_dir, 'paymentmethod_distribution.png'))
        plt.close()

        # Payment Method by Gender
        plt.figure(figsize=(8,6))
        sns.countplot(x='PaymentMethod', hue='gender', data=df, palette=['#4169E1', '#E74C3C'])
        plt.title('Payment Method by Gender')
        plt.savefig(os.path.join(vis_dir, 'paymentmethod_gender.png'))
        plt.close()

        # Contract Type Distribution
        plt.figure()
        counts = df['Contract'].value_counts()
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=colors_3)
        plt.title('Contract Type Distribution')
        plt.savefig(os.path.join(vis_dir, 'contract_distribution.png'))
        plt.close()

        # Contract by Partner status
        plt.figure(figsize=(10,6))
        sns.countplot(x='Contract', hue='Partner', data=df, palette=['#4169E1', '#E74C3C'])
        plt.title('Contract Type by Partner Status')
        plt.savefig(os.path.join(vis_dir, 'contract_demographics.png'))
        plt.close()

        # Payment Method by Gender
        plt.figure(figsize=(8,6))
        sns.countplot(x='PaymentMethod', hue='gender', data=df, palette=['#4169E1', '#E74C3C'])
        plt.title('Payment Method by Gender')
        plt.savefig(os.path.join(vis_dir, 'payment_method_gender.png'))
        plt.close()

        print("Payment and contract visualizations saved.")
    except Exception as e:
        logging.error(f"Error in payment and contract analysis: {str(e)}")
        raise


def churn_related_analysis(df, vis_dir):
    """Visualizations related directly to churn and some distributions."""
    print("\nPerforming churn-related analysis...")
    try:
        pie_colors_2 = ['#0600b0', '#e30517']  # Royal blue, red
        pie_colors_3 = ['#0600b0', '#e30517', '#e805b3']

        # Churn Distribution
        plt.figure()
        df['Churn'].value_counts().plot.pie(autopct='%1.1f%%', colors=pie_colors_2)
        plt.title('Churn Distribution')
        plt.ylabel('')
        plt.savefig(os.path.join(vis_dir, 'churn_distribution.png'))
        plt.close()

        # Gender vs Churn
        plt.figure(figsize=(6, 4))
        sns.countplot(x='gender', hue='Churn', data=df, palette=pie_colors_2)
        plt.title('Gender vs Churn')
        plt.savefig(os.path.join(vis_dir, 'gender_churn.png'))
        plt.close()

        # Senior Citizen vs Churn
        plt.figure(figsize=(6, 4))
        sns.countplot(x='SeniorCitizen', hue='Churn', data=df, palette=pie_colors_2)
        plt.title('Senior Citizen vs Churn')
        plt.savefig(os.path.join(vis_dir, 'seniorcitizen_churn.png'))
        plt.close()

        # Payment Method vs Churn
        plt.figure(figsize=(8, 4))
        sns.countplot(x='PaymentMethod', hue='Churn', data=df, palette=['#4169E1', '#E74C3C', '#FF69B4', '#2ECC71'])
        plt.title('Payment Method vs Churn')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'paymentmethod_churn.png'))
        plt.close()

        # Partner and Dependents vs Churn (side by side)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.countplot(x='Partner', hue='Churn', data=df, ax=axes[0], palette=pie_colors_2)
        axes[0].set_title('Partner vs Churn')
        sns.countplot(x='Dependents', hue='Churn', data=df, ax=axes[1], palette=pie_colors_2)
        axes[1].set_title('Dependents vs Churn')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'partner_dependents_churn.png'))
        plt.close()

        # Tenure Distribution by Churn
        plt.figure(figsize=(8, 5))
        sns.histplot(x='tenure', hue='Churn', data=df, multiple='stack', bins=30, palette=pie_colors_2)
        plt.title('Tenure Distribution by Churn')
        plt.xlabel('Tenure (Months)')
        plt.ylabel('Number of Customers')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'tenure_distribution.png'))
        plt.close()

        # Monthly Charges Distribution by Churn (KDE)
        plt.figure(figsize=(8, 5))
        sns.kdeplot(x='MonthlyCharges', hue='Churn', data=df, fill=True, common_norm=False, palette=pie_colors_2)
        plt.title('Monthly Charges Distribution by Churn')
        plt.xlabel('Monthly Charges')
        plt.ylabel('Density')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'monthlycharges_churn_kde.png'))
        plt.close()

        # Total Charges Distribution by Churn (KDE)
        plt.figure(figsize=(8, 5))
        sns.kdeplot(x='TotalCharges', hue='Churn', data=df, fill=True, common_norm=False, palette=pie_colors_2)
        plt.title('Total Charges Distribution by Churn')
        plt.xlabel('Total Charges')
        plt.ylabel('Density')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'totalcharges_churn_kde.png'))
        plt.close()

        # Violin Plots: Monthly Charges & Tenure by Churn
        plt.figure(figsize=(7, 5))
        sns.violinplot(x='Churn', y='MonthlyCharges', data=df, palette=pie_colors_2)
        plt.title('Monthly Charges by Churn (Violin Plot)')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'monthlycharges_violin.png'))
        plt.close()

        plt.figure(figsize=(7, 5))
        sns.violinplot(x='Churn', y='tenure', data=df, palette=pie_colors_2)
        plt.title('Tenure by Churn (Violin Plot)')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'tenure_violin.png'))
        plt.close()

        print("Churn-related visualizations saved.")
    except Exception as e:
        logging.error(f"Error in churn-related analysis: {str(e)}")
        raise


def other_distributions(df, vis_dir):
    """Other distributions for categorical features."""
    print("\nPerforming other distributions analysis...")
    try:
        # Monthly Charges Distribution (Histogram + KDE)
        plt.figure(figsize=(8,6))
        sns.histplot(df['MonthlyCharges'], kde=True, color='#4169E1')
        plt.title('Monthly Charges Distribution')
        plt.savefig(os.path.join(vis_dir, 'monthlycharges_distribution.png'))
        plt.close()

        # Total Charges Distribution (Histogram + KDE)
        plt.figure(figsize=(8,6))
        sns.histplot(df['TotalCharges'], kde=True, color='#E74C3C')
        plt.title('Total Charges Distribution')
        plt.savefig(os.path.join(vis_dir, 'totalcharges_distribution.png'))
        plt.close()

        # Multiple Lines Distribution (Pie)
        plt.figure()
        counts = df['MultipleLines'].value_counts()
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=['#4169E1', '#E74C3C', '#FF69B4'])
        plt.title('Multiple Lines Distribution')
        plt.savefig(os.path.join(vis_dir, 'multiplelines_distribution.png'))
        plt.close()

        # Paperless Billing Distribution (Pie)
        plt.figure()
        counts = df['PaperlessBilling'].value_counts()
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=['#4169E1', '#E74C3C'])
        plt.title('Paperless Billing Distribution')
        plt.savefig(os.path.join(vis_dir, 'paperlessbilling_distribution.png'))
        plt.close()

        # Partner and Dependents Distribution (Countplot with hue)
        plt.figure()
        sns.countplot(x='Partner', hue='Dependents', data=df, palette=['#4169E1', '#E74C3C'])
        plt.title('Partner and Dependents Distribution')
        plt.savefig(os.path.join(vis_dir, 'partner_dependents_distribution.png'))
        plt.close()

        # Phone Service Distribution (Pie)
        plt.figure()
        counts = df['PhoneService'].value_counts()
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=['#4169E1', '#E74C3C'])
        plt.title('Phone Service Distribution')
        plt.savefig(os.path.join(vis_dir, 'phoneservice_distribution.png'))
        plt.close()

        # Streaming TV Distribution (Pie)
        plt.figure()
        counts = df['StreamingTV'].value_counts()
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=['#4169E1', '#E74C3C'])
        plt.title('Streaming TV Distribution')
        plt.savefig(os.path.join(vis_dir, 'streamingtv_distribution.png'))
        plt.close()

        # Streaming Movies Distribution (Pie)
        plt.figure()
        counts = df['StreamingMovies'].value_counts()
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=['#4169E1', '#E74C3C'])
        plt.title('Streaming Movies Distribution')
        plt.savefig(os.path.join(vis_dir, 'streamingmovies_distribution.png'))
        plt.close()

        # Tech Support Distribution (Pie)
        plt.figure()
        counts = df['TechSupport'].value_counts()
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=['#4169E1', '#E74C3C'])
        plt.title('Tech Support Distribution')
        plt.savefig(os.path.join(vis_dir, 'techsupport_distribution.png'))
        plt.close()

        # Device Protection Distribution (Pie)
        plt.figure()
        counts = df['DeviceProtection'].value_counts()
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=['#4169E1', '#E74C3C'])
        plt.title('Device Protection Distribution')
        plt.savefig(os.path.join(vis_dir, 'deviceprotection_distribution.png'))
        plt.close()

        print("Other distributions saved.")
    except Exception as e:
        logging.error(f"Error in other distributions analysis: {str(e)}")
        raise


def prepare_features(df):
    """Prepare features and target variable using SelectKBest for feature selection."""
    print("\n2. Preparing features and target variable...")
    try:
        # Map Churn to binary (Yes=1, No=0)
        y = df['Churn'].map({'Yes': 1, 'No': 0})
        
        # Columns to drop (only drop if they exist)
        columns_to_drop = ['customerID', 'Churn', 'Race', 'Gender']
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        X = df.drop(existing_columns_to_drop, axis=1)
        
        # One-hot encoding for categorical features
        X_encoded = pd.get_dummies(X)
        print(f"Number of features after encoding: {X_encoded.shape[1]}")
        
        # Drop low-variance columns (columns with only 1 unique value)
        low_variance_cols = [col for col in X_encoded.columns if X_encoded[col].nunique() == 1]
        X_encoded = X_encoded.drop(columns=low_variance_cols)
        print(f"Number of features after dropping low-variance columns: {X_encoded.shape[1]}")

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42
        )

        # Feature selection with SelectKBest (using chi2)
        print("\nSelecting top features with SelectKBest...")
        selector = SelectKBest(score_func=chi2, k=20)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # Get selected feature names
        selected_feature_names = X_train.columns[selector.get_support()].tolist()
        print(f"Top selected features: {selected_feature_names}")

        # Apply SMOTE on selected features to balance classes
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)

        # **Important: Convert resampled numpy array back to DataFrame with column names**
        X_train_resampled_df = pd.DataFrame(X_train_resampled, columns=selected_feature_names)

        # Return DataFrame for train, numpy array for test (or convert test similarly if needed)
        return X_train_resampled_df, X_test_selected, y_train_resampled, y_test, selected_feature_names

    except Exception as e:
        logging.error(f"Error preparing features: {str(e)}")
        raise


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
            'kernel': ['linear'],
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
        # Parse command-line arguments for flexible data path
        parser = argparse.ArgumentParser(description="Run Telco churn ML pipeline")
        parser.add_argument(
            '--data-path',
            default=None,
            help='Path to the dataset CSV file'
        )
        args = parser.parse_args()

        # Try multiple default paths including user input
        possible_paths = [
            args.data_path,
            'uploads/telcodataset.csv',
            '/Users/user/Downloads/telcodataset.csv'
        ]
        data_path = None
        for p in possible_paths:
            if p and os.path.isfile(p):
                data_path = p
                break

        if data_path is None:
            raise FileNotFoundError("Dataset CSV file not found in provided or default locations.")

        print(f"Using dataset path: {data_path}")

        # Create results directory and initialize database
        results_dir = create_results_directory()
        initialize_database()

        # Load and preprocess data
        df = load_and_preprocess_data(data_path)

        # Create the directory to save visualizations
        base_dir = os.path.dirname(os.path.abspath(__file__))  # path to ml_analysis.py folder
        vis_dir = os.path.join(base_dir, 'visualizations')

        os.makedirs(vis_dir, exist_ok=True)

        # Call the visualization functions
        demographic_analysis(df, vis_dir)
        correlation_analysis(df, vis_dir)
        internet_service_analysis(df, vis_dir)
        service_usage_analysis(df, vis_dir)
        payment_and_contract_analysis(df, vis_dir)
        churn_related_analysis(df, vis_dir)
        other_distributions(df, vis_dir)
        additional_services_churn(df, vis_dir)
        online_backup_distribution(df, vis_dir)
        online_security_distribution(df, vis_dir) 
        internet_service_senior(df, vis_dir)

        # Prepare features
        X_train, X_test, y_train, y_test, selected_feature_names = prepare_features(df)
        print("Selected features:", selected_feature_names)


        # Train models
        model_results, trained_models = train_models(X_train, X_test, y_train, y_test)

        # Save trained models to disk for future use
        save_models(trained_models)

        # Create ensemble from the trained models (use trained_models, not new instances)
        ensemble_accuracy, ensemble_report = create_ensemble(
            X_train, X_test, y_train, y_test, model_results, trained_models
        )

        # Plot ROC curves using trained models
        plot_roc_curves(trained_models, X_test, y_test, results_dir)

        # Plot and save performance comparison visuals
        plot_results(model_results, results_dir)

        # Save text report of results
        save_results(model_results, ensemble_accuracy, ensemble_report, results_dir)

        # Example: take first test sample, predict with best model, and save to DB
        best_model_name = max(model_results, key=lambda k: model_results[k]['accuracy'])
        best_model = trained_models[best_model_name]

        sample_input = X_test[0]  # numpy array, scaled and selected features
        sample_input_list = sample_input.tolist()

        prediction = best_model.predict(sample_input.reshape(1, -1))[0]

        # TODO: Provide correct column names corresponding to selected features here.
        # For now, placeholders:
        column_names = [f"feature_{i}" for i in range(len(sample_input_list))]

        save_prediction(sample_input_list, int(prediction), best_model_name, column_names)

        print("\nAnalysis completed successfully!")

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        print(f"Error in main execution. Check error_log.txt for details.")
        raise


if __name__ == "__main__":
    main()
