# Telco Customer Churn Analysis

This project performs a comprehensive analysis of customer churn in a telecommunications company using the Telco Customer Churn dataset from Kaggle.

## Dataset Information

The dataset contains information about 7,043 customers with 21 distinct features, including:
- Demographic information (gender, senior citizen status, partner status, dependents)
- Service information (internet service type, additional services)
- Account information (tenure, contract type, payment method, charges)
- Target variable: Customer churn status (Yes/No)

Dataset source: [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Project Structure

```
sdp/
├── data_analysis.py      # Main analysis script
├── visualizations/       # Generated visualization files
├── requirements.txt      # Required Python packages
└── README.md            # Project documentation
```

## Analysis Components

The analysis includes:

1. **Basic Dataset Information**
   - Dataset dimensions
   - Feature descriptions
   - Data types

2. **Demographic Analysis**
   - Gender distribution
   - Senior citizen analysis
   - Partner and dependents status

3. **Service Usage Analysis**
   - Internet service types
   - Additional services
   - Service bundle patterns

4. **Payment Analysis**
   - Payment method preferences
   - Contract type analysis
   - Billing preferences

5. **Churn Analysis**
   - Overall churn rate
   - Demographic factors affecting churn
   - Service-related churn factors
   - Payment method impact on churn

6. **Numerical Analysis**
   - Monthly charges
   - Total charges
   - Tenure analysis
   - Correlation analysis

## Requirements

To run this analysis, you need the following Python packages:
```
pandas
numpy
seaborn
matplotlib
```

## How to Run

1. Install the required packages using either pip or pip3:
```bash
# Using pip
pip install -r requirements.txt

# Or using pip3
pip3 install -r requirements.txt
```

2. Download the dataset from Kaggle and place it in your working directory

3. Run the analysis script:
```bash
python data_analysis.py
```

## Output

The script generates:
- Terminal output with statistical analysis results
- Visualizations saved in the `visualizations/` directory:
  - Distribution plots
  - Correlation heatmaps
  - Demographic analysis charts
  - Service usage analysis
  - Churn analysis visualizations

## Key Findings

The analysis reveals important insights about:
- Customer demographics and their impact on churn
- Service preferences and their relationship to customer retention
- Payment method preferences across different customer segments
- Key factors influencing customer churn

## Visualization Files

The `visualizations/` directory contains the following types of charts:
- Distribution plots for all features
- Correlation analysis heatmaps
- Demographic analysis charts
- Service usage analysis
- Churn analysis visualizations
- Payment method analysis
- Contract type analysis

## Contributing

Feel free to contribute to this project by:
- Suggesting additional analyses
- Improving visualization techniques
- Adding machine learning models
- Enhancing documentation

## License

This project is open source and available under the MIT License. 
