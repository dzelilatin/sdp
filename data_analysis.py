print("Script started")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import re
import os

# Create visualizations directory if it doesn't exist
visualizations_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualizations')
if not os.path.exists(visualizations_dir):
    os.makedirs(visualizations_dir)

# Set up visualization parameters
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Load the dataset
print("\n1. Loading Dataset...")
df = pd.read_csv('/Users/user/Downloads/telcodataset.csv')

# Data Preprocessing
print("\n2. Data Preprocessing...")

# Debug TotalCharges column
print("\nDebugging TotalCharges column:")
print("First few values:", df['TotalCharges'].head())
print("Data type:", df['TotalCharges'].dtype)
print("Unique values:", df['TotalCharges'].unique()[:10])

# Create a new TotalCharges column based on MonthlyCharges * tenure
print("\nCreating new TotalCharges column...")
df['TotalCharges'] = df['MonthlyCharges'] * df['tenure']

# Convert SeniorCitizen to categorical for better analysis
df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})

# Verify data types
print("\nVerifying data types after preprocessing:")
print(df.dtypes)

# 1. Basic Dataset Information
print("\n3. Basic Dataset Information:")
print(f"Number of rows (samples): {df.shape[0]}")
print(f"Number of columns (features): {df.shape[1]}")
print("\nColumn Information:")
print(df.info())  

# 2. Missing Values Analysis
print("\n4. Missing Values Analysis:")
missing_data = df.isnull().sum()
print(missing_data[missing_data > 0] if missing_data.any() else "No missing values found!")

# 3. Text Analysis
print("\n5. Text Analysis:")
text_columns = df.select_dtypes(include=['object']).columns
for col in text_columns:
    if col != 'customerID':  # Skip customer ID as it's unique
        print(f"\nAnalysis for {col}:")
        # Calculate text lengths
        text_lengths = df[col].astype(str).apply(len)
        print(f"Average length: {text_lengths.mean():.2f}")
        print(f"Min length: {text_lengths.min()}")
        print(f"Max length: {text_lengths.max()}")
        
        # For columns that might contain sentences
        if any(' ' in str(x) for x in df[col].head()):
            word_counts = df[col].astype(str).apply(lambda x: len(x.split()))
            print(f"Average words per entry: {word_counts.mean():.2f}")

# 4. Numerical Analysis
print("\n6. Numerical Analysis:")
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_cols:
    print(f"\nAnalysis for {col}:")
    print(df[col].describe())
    
    # Create distribution plot for each numerical column
    plt.figure()
    sns.histplot(data=df, x=col, kde=True)
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(visualizations_dir, f'{col}_distribution.png'))
    plt.close()

# 5. Categorical Analysis
print("\n7. Categorical Analysis:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col != 'customerID':  # Skip customer ID
        print(f"\nAnalysis for {col}:")
        value_counts = df[col].value_counts()
        print(value_counts)
        
        # Create bar plot for each categorical column
        plt.figure()
        sns.countplot(data=df, x=col)
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(visualizations_dir, f'{col}_distribution.png'))
        plt.close()

# 6. Target Variable Analysis (Churn)
print("\n8. Target Variable Analysis (Churn):")
churn_distribution = df['Churn'].value_counts()
print(churn_distribution)
print(f"\nChurn Rate: {(churn_distribution['Yes'] / len(df) * 100):.2f}%")

# Create churn distribution plot
plt.figure()
sns.countplot(data=df, x='Churn')
plt.title('Distribution of Churn')
plt.tight_layout()
plt.savefig(os.path.join(visualizations_dir, 'churn_distribution.png'))
plt.close()

# 7. Correlation Analysis
print("\n9. Correlation Analysis:")
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
correlation_matrix = df[numerical_cols].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Create correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.savefig(os.path.join(visualizations_dir, 'correlation_matrix.png'))
plt.close()

# 8. Feature Importance Analysis
print("\n10. Feature Importance Analysis:")
# Convert categorical variables to numerical for analysis
df_encoded = pd.get_dummies(df.drop(['customerID', 'Churn'], axis=1))
df_encoded['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Calculate correlation with target
correlations = df_encoded.corr()['Churn'].sort_values(ascending=False)
print("\nTop 10 features most correlated with Churn:")
print(correlations[1:11])  # Skip Churn itself

# 9. Demographic Analysis
print("\n11. Demographic Analysis:")
print("\nGender Distribution:")
print(df['gender'].value_counts())
print("\nSenior Citizen Distribution:")
print(df['SeniorCitizen'].value_counts())
print("\nPartner Distribution:")
print(df['Partner'].value_counts())
print("\nDependents Distribution:")
print(df['Dependents'].value_counts())

# Gender Distribution
plt.figure()
gender_counts = df['gender'].value_counts()
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
plt.title('Gender Distribution')
plt.savefig(os.path.join(visualizations_dir, 'gender_distribution.png'))
plt.close()

# Senior Citizen Distribution
plt.figure()
senior_counts = df['SeniorCitizen'].value_counts()
plt.pie(senior_counts, labels=senior_counts.index, autopct='%1.1f%%')
plt.title('Senior Citizen Distribution')
plt.savefig(os.path.join(visualizations_dir, 'senior_citizen_distribution.png'))
plt.close()

# Partner and Dependents Distribution
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.countplot(data=df, x='Partner', ax=axes[0])
axes[0].set_title('Partner Distribution')
sns.countplot(data=df, x='Dependents', ax=axes[1])
axes[1].set_title('Dependents Distribution')
plt.tight_layout()
plt.savefig(os.path.join(visualizations_dir, 'partner_dependents_distribution.png'))
plt.close()

# 10. Service Usage Analysis
print("\n12. Service Usage Analysis:")
print("\nInternet Service Distribution:")
print(df['InternetService'].value_counts())
print("\nAdditional Services Distribution:")
additional_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                      'TechSupport', 'StreamingTV', 'StreamingMovies']
for service in additional_services:
    print(f"\n{service}:")
    print(df[service].value_counts())

# Internet Service Distribution
plt.figure()
internet_counts = df['InternetService'].value_counts()
plt.pie(internet_counts, labels=internet_counts.index, autopct='%1.1f%%')
plt.title('Internet Service Distribution')
plt.savefig(os.path.join(visualizations_dir, 'internet_service_distribution.png'))
plt.close()

# Additional Services Analysis
additional_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                      'TechSupport', 'StreamingTV', 'StreamingMovies']
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for i, service in enumerate(additional_services):
    row = i // 3
    col = i % 3
    sns.countplot(data=df, x=service, ax=axes[row, col])
    axes[row, col].set_title(f'{service} Distribution')
plt.tight_layout()
plt.savefig(os.path.join(visualizations_dir, 'additional_services_distribution.png'))
plt.close()

# 11. Payment Analysis
print("\n13. Payment Analysis:")
print("\nPayment Method Distribution:")
print(df['PaymentMethod'].value_counts())
print("\nContract Type Distribution:")
print(df['Contract'].value_counts())

# Payment Method Distribution
plt.figure()
payment_counts = df['PaymentMethod'].value_counts()
plt.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%')
plt.title('Payment Method Distribution')
plt.savefig(os.path.join(visualizations_dir, 'payment_method_distribution.png'))
plt.close()

# 12. Demographic vs Churn Analysis
print("\n14. Demographic vs Churn Analysis:")
print("\nGender vs Churn:")
print(pd.crosstab(df['gender'], df['Churn']))
print("\nSenior Citizen vs Churn:")
print(pd.crosstab(df['SeniorCitizen'], df['Churn']))
print("\nPartner vs Churn:")
print(pd.crosstab(df['Partner'], df['Churn']))
print("\nDependents vs Churn:")
print(pd.crosstab(df['Dependents'], df['Churn']))

# Gender vs Churn
plt.figure()
sns.countplot(data=df, x='gender', hue='Churn')
plt.title('Gender Distribution by Churn Status')
plt.savefig(os.path.join(visualizations_dir, 'gender_churn.png'))
plt.close()

# Senior Citizen vs Churn
plt.figure()
sns.countplot(data=df, x='SeniorCitizen', hue='Churn')
plt.title('Senior Citizen Distribution by Churn Status')
plt.savefig(os.path.join(visualizations_dir, 'senior_citizen_churn.png'))
plt.close()

# Partner and Dependents vs Churn
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.countplot(data=df, x='Partner', hue='Churn', ax=axes[0])
axes[0].set_title('Partner Distribution by Churn Status')
sns.countplot(data=df, x='Dependents', hue='Churn', ax=axes[1])
axes[1].set_title('Dependents Distribution by Churn Status')
plt.tight_layout()
plt.savefig(os.path.join(visualizations_dir, 'partner_dependents_churn.png'))
plt.close()

# 13. Service Usage vs Churn Analysis
print("\n15. Service Usage vs Churn Analysis:")
print("\nInternet Service vs Churn:")
print(pd.crosstab(df['InternetService'], df['Churn']))
print("\nAdditional Services vs Churn:")
for service in additional_services:
    print(f"\n{service} vs Churn:")
    print(pd.crosstab(df[service], df['Churn']))

# Internet Service vs Churn
plt.figure()
sns.countplot(data=df, x='InternetService', hue='Churn')
plt.title('Internet Service Distribution by Churn Status')
plt.savefig(os.path.join(visualizations_dir, 'internet_service_churn.png'))
plt.close()

# Additional Services vs Churn
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for i, service in enumerate(additional_services):
    row = i // 3
    col = i % 3
    sns.countplot(data=df, x=service, hue='Churn', ax=axes[row, col])
    axes[row, col].set_title(f'{service} Distribution by Churn Status')
plt.tight_layout()
plt.savefig(os.path.join(visualizations_dir, 'additional_services_churn.png'))
plt.close()

# 14. Payment Method vs Churn Analysis
print("\nPayment Method vs Churn:")
print(pd.crosstab(df['PaymentMethod'], df['Churn']))

plt.figure()
sns.countplot(data=df, x='PaymentMethod', hue='Churn')
plt.title('Payment Method Distribution by Churn Status')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(visualizations_dir, 'payment_method_churn.png'))
plt.close()

# 15. Demographic Preferences Analysis
print("\n16. Demographic Preferences Analysis:")
print("\nPayment Method Preferences by Gender:")
print(pd.crosstab(df['PaymentMethod'], df['gender']))
print("\nInternet Service Preferences by Senior Citizen Status:")
print(pd.crosstab(df['InternetService'], df['SeniorCitizen']))
print("\nContract Type Preferences by Gender:")
print(pd.crosstab(df['Contract'], df['gender']))
print("\nContract Type Preferences by Senior Citizen Status:")
print(pd.crosstab(df['Contract'], df['SeniorCitizen']))
print("\nContract Type Preferences by Partner Status:")
print(pd.crosstab(df['Contract'], df['Partner']))
print("\nContract Type Preferences by Dependents Status:")
print(pd.crosstab(df['Contract'], df['Dependents']))

# Payment Method Preferences by Gender
plt.figure()
sns.countplot(data=df, x='PaymentMethod', hue='gender')
plt.title('Payment Method Preferences by Gender')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(visualizations_dir, 'payment_method_gender.png'))
plt.close()

# Internet Service Preferences by Senior Citizen Status
plt.figure()
sns.countplot(data=df, x='InternetService', hue='SeniorCitizen')
plt.title('Internet Service Preferences by Senior Citizen Status')
plt.savefig(os.path.join(visualizations_dir, 'internet_service_senior.png'))
plt.close()

# Contract Type Preferences by Demographic
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
sns.countplot(data=df, x='Contract', hue='gender', ax=axes[0, 0])
axes[0, 0].set_title('Contract Type by Gender')
sns.countplot(data=df, x='Contract', hue='SeniorCitizen', ax=axes[0, 1])
axes[0, 1].set_title('Contract Type by Senior Citizen Status')
sns.countplot(data=df, x='Contract', hue='Partner', ax=axes[1, 0])
axes[1, 0].set_title('Contract Type by Partner Status')
sns.countplot(data=df, x='Contract', hue='Dependents', ax=axes[1, 1])
axes[1, 1].set_title('Contract Type by Dependents Status')
plt.tight_layout()
plt.savefig(os.path.join(visualizations_dir, 'contract_demographics.png'))
plt.close()

print(f"\nAnalysis complete! All visualizations have been saved in the '{visualizations_dir}' directory.") 