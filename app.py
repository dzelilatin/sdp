import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from ml_analysis import (
    add_age_range,
    create_results_directory,
    initialize_database,
    prepare_features,
    train_models,
    create_ensemble,
    plot_roc_curves,
    plot_results,
    save_prediction,
)

# Set up Streamlit app
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📉 Customer Churn Prediction Dashboard")

uploaded_file = st.file_uploader("Upload your customer data CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Uploaded Data")
    st.write(df.head())

    # Setup directories and DB
    create_results_directory()
    initialize_database()

    # Feature preparation
    df = add_age_range(df)
    X_train_resampled_df, X_test_selected, y_train_resampled, y_test, selected_feature_names = prepare_features(df)
    selected_X_train = X_train_resampled_df

    st.success("✅ Data prepared and features extracted.")

    # Model training
    st.subheader("Training Models...")
    model_results, trained_models = train_models(
        selected_X_train, X_test_selected, y_train_resampled, y_test
    )

    # Basic checks
    assert isinstance(model_results, dict), "Expected model_results to be a dict"
    assert isinstance(trained_models, dict), "Expected trained_models to be a dict"

    st.success("✅ Models trained.")

    # Create ensemble model
    ensemble_model = create_ensemble(
        X_train=selected_X_train,
        X_test=X_test_selected,
        y_train=y_train_resampled,
        y_test=y_test,
        model_results=model_results,
        models=trained_models
    )
    st.success("✅ Ensemble model created.")

    # Plotting results
    st.subheader("📈 Model Performance")

    plot_roc_curves(trained_models, X_test_selected, y_test, results_dir="ml_results")
    st.image("ml_results/roc_curves.png")

    plot_results(model_results, results_dir="ml_results")
    st.image("ml_results/model_accuracy_comparison.png")
    st.image("ml_results/model_metrics_comparison.png")

    # Prediction interface
    st.subheader("📊 Predict New Customer Churn")
    input_data = {}

    for col in selected_X_train.columns:
        val = st.text_input(f"{col}", "")
        input_data[col] = val

    if st.button("🔍 Predict Churn"):
        try:
            input_df = pd.DataFrame([input_data])
            input_df = input_df.astype(selected_X_train.dtypes.to_dict())  # Match types

            prediction = ensemble_model.predict(input_df)[0]
            proba = ensemble_model.predict_proba(input_df)[0][1]

            st.success(f"🎯 Prediction: {'Churn' if prediction == 1 else 'Not Churn'}")
            st.info(f"🧠 Probability of Churn: {proba:.2f}")

            # Save prediction
            save_prediction(input_data, prediction, proba)
            st.success("✅ Prediction saved successfully.")

        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
