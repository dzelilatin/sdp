# Predicting Customer Churn Using Machine Learning: A Data-Driven Approach for Retention Strategies

## Overview

In today’s competitive market, customer retention is critical for business success. This project aims to predict customer churn using machine learning techniques, enabling businesses to identify at-risk customers and implement strategies to retain them. By analyzing the Telco Customer Churn dataset, this project provides actionable insights into churn factors and customer behavior.

## Dataset Information

The dataset contains information about 7,043 customers with 21 features, including:
- **Demographics**: Gender, senior citizen status, partner status, dependents.
- **Services**: Internet service type, additional services (e.g., security, backups).
- **Account Details**: Tenure, contract type, payment method, monthly charges, total charges.
- **Target Variable**: Customer churn status (Yes/No).

Dataset source: [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Project Objectives

1. **Predict Customer Churn**: Build machine learning models to classify customers as likely to churn or not.
2. **Analyze Churn Factors**: Identify key features influencing churn (e.g., demographics, services, payment methods).
3. **Provide Actionable Insights**: Help businesses reduce churn through targeted strategies.
4. **Visualize Results**: Present findings in an accessible format for stakeholders.

## Project Structure

```
sdp/
├── app.py                # Flask application for user interaction
├── ml_analysis.py        # Machine learning pipeline
├── data_analysis.py      # Exploratory data analysis
├── visualizations/       # Generated visualization files
├── requirements.txt      # Required Python packages
└── README.md             # Project documentation
```

## Features and Analysis Components

### **1. Exploratory Data Analysis**
- **Demographics**: Gender distribution, senior citizen analysis, partner and dependents status.
- **Service Usage**: Internet service types, additional services, service bundle patterns.
- **Payment Analysis**: Payment method preferences, contract type analysis, billing preferences.
- **Churn Analysis**: Overall churn rate, demographic factors affecting churn, service-related churn factors, payment method impact on churn.
- **Numerical Analysis**: Monthly charges, total charges, tenure analysis, correlation analysis.

### **2. Machine Learning Pipeline**
- **Preprocessing**: One-hot encoding, scaling, and feature selection using Recursive Feature Elimination (RFE).
- **Model Training**: Logistic Regression, Random Forest, XGBoost, SVM, Gradient Boosting, AdaBoost, Neural Networks.
- **Hyperparameter Tuning**: Optimize model parameters using GridSearchCV.
- **Evaluation Metrics**: Accuracy, precision, recall, F1 score, AUC-ROC.

### **3. Visualization**
- **Churn Trends**: Distribution plots, correlation heatmaps, demographic analysis charts.
- **Model Performance**: ROC curves, comparison of accuracy, precision, recall, and false positives.

### **4. Actionable Insights**
- Senior citizens are more likely to churn—target retention strategies for this group.
- Customers with higher monthly charges are at risk—offer discounts or loyalty programs.
- Fiber optic users show dissatisfaction—improve service quality or pricing.
- Bundling additional services like security and backups reduces churn.

## Requirements

To run this project, you need the following Python packages:
```
pandas
numpy
seaborn
matplotlib
scikit-learn
xgboost
flask
```

## How to Run

### **1. Install Dependencies**
Install the required packages using pip:
```bash
pip install -r requirements.txt
```

### **2. Run the Flask Application**
Start the application to interact with the project:
```bash
python app.py
```

### **3. Upload Dataset**
Upload the Telco Customer Churn dataset (CSV format) through the web interface.

### **4. View Results**
- Visualize churn trends and model performance.
- Download generated plots and results.

## Output

The project generates:
- **Terminal Output**: Statistical analysis results and actionable insights.
- **Visualizations**: Saved in the `visualizations/` directory:
  - Distribution plots
  - Correlation heatmaps
  - Demographic analysis charts
  - Service usage analysis
  - Churn analysis visualizations
- **Model Results**: Saved in the `ml_results/` directory:
  - Model performance metrics
  - Ensemble model results

## Contributing

Contributions are welcome! You can:
- Suggest additional analyses.
- Improve visualization techniques.
- Add new machine learning models.
- Enhance documentation.

## Ethical Considerations

This project ensures:
- **Data Privacy**: Sensitive features like race and gender are excluded to prevent bias.
- **Fairness**: Models are evaluated to avoid discrimination against specific customer groups.

## License

This project is open source and available under the MIT License.

---

## Contact

For questions or feedback, feel free to reach out at [dzelila.tinjak@stu.ibu.edu.ba].

## Quick Demo (Load-only Mode)

1. Train once with your dataset and let the app save `ml_results/final_pipeline.joblib`.
2. Next runs: check "Load saved pipeline (skip training)" in the sidebar to start instantly.

## Docker (optional)

Create `Dockerfile` like:

```Dockerfile
# Simple Streamlit runtime
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
```

Build and run:
```bash
docker build -t churn-app .
docker run -p 8501:8501 churn-app
```