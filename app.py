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
    save_prediction_streamlit,
    transform_new_data,
    top_permutation_importances,
    get_prediction_history,
    find_best_threshold,
    save_full_pipeline,
    export_model_results,
    load_full_pipeline,
)
import shap
import numpy as np
import json
import os

# Caching helpers
@st.cache_resource
def cached_load_pipeline(path):
    try:
        from ml_analysis import load_full_pipeline as load_fn
        return load_fn(path)
    except Exception as e:
        raise e

@st.cache_data
def cached_export_results(model_results):
    return model_results

# Set up Streamlit app
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📉 Customer Churn Prediction Dashboard")

# Sidebar: training controls
with st.sidebar:
    st.header("Training Controls")
    
    use_saved = st.checkbox("Load saved pipeline (skip training)", value=False)
    st.caption("Check this to quickly make predictions using the last saved model.")

    pipeline_path = st.text_input("Pipeline path", value="ml_results/final_pipeline.joblib")
    
    cv = st.number_input("Cross-validation folds", min_value=2, max_value=10, value=3)
    st.caption("Number of rounds for model validation. Higher is more robust but slower.")

    randomized = st.checkbox("Use RandomizedSearchCV", value=True)
    st.caption("Check for faster training by testing random settings. Uncheck for a slower but more thorough search.")

    best_only = st.checkbox("Train best models only (faster)", value=True)
    st.caption("Check to train only a small subset of the fastest models. Uncheck to compare all model types.")

    use_smote_in_cv = st.checkbox("Use SMOTE inside CV", value=True)
    st.caption("Recommended. Helps prevent model bias by balancing the 'Churn' vs 'No Churn' data.")

    calibrate = st.checkbox("Calibrate probabilities", value=True)
    st.caption("Recommended. Makes the 'Risk Score' percentage more reliable.")

    overwrite_after_train = st.checkbox("Re-train and overwrite saved pipeline", value=True)
    st.caption("Check to save the newly trained model, overwriting the old one.")

uploaded_file = st.file_uploader("Upload your customer data CSV file", type=["csv"], key="upload")

# Validation helpers
EXPECTED_COLUMNS = None

def validate_schema(df):
    missing = []
    if EXPECTED_COLUMNS is not None:
        missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    return missing

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    best_t = 0.5
    st.subheader("Raw Uploaded Data (head)")
    st.write(df.head())

    # Normalize SeniorCitizen to 'Yes'/'No' for consistency
    if 'SeniorCitizen' in df.columns:
        df['SeniorCitizen'] = df['SeniorCitizen'].apply(lambda x: 'Yes' if str(x).strip().lower() in ['1', 'yes', 'true'] else 'No')

    # Optional schema validation display
    if EXPECTED_COLUMNS is not None:
        missing = validate_schema(df)
        if missing:
            st.warning(f"Missing expected columns: {missing}")

    # Sanitize categories: convert unexpected levels to string and let OHE ignore
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).fillna("Unknown")

    # Setup directories and DB
    create_results_directory()
    initialize_database()

    # Feature preparation
    df = add_age_range(df)
    X_train_resampled_df, X_test_selected, y_train_resampled, y_test, selected_feature_names, preprocessor, selector, X_train_selected_df_pre_smote, y_train_pre = prepare_features(df)
    selected_X_train = X_train_resampled_df

    EXPECTED_COLUMNS = [c for c in df.columns if c != 'Churn']

    st.success("✅ Data prepared and features extracted.")

    if use_saved:
        try:
            artifact = cached_load_pipeline(pipeline_path)
            preprocessor = artifact['preprocessor']
            selector = artifact['selector']
            ensemble_model = artifact['model']
            best_t = artifact.get('threshold', 0.5)
            expected_cols_saved = artifact.get('expected_columns')
            ensemble_accuracy = np.nan
            ensemble_report = f"Loaded pipeline: {pipeline_path}"
            model_results = {}
            trained_models = {}
            st.success(f"✅ Loaded saved pipeline from {pipeline_path}.")
            # Validate schema if artifact has expected columns
            if expected_cols_saved is not None:
                missing = [c for c in expected_cols_saved if c not in df.columns]
                if missing:
                    st.error(f"Uploaded CSV is missing required columns for this pipeline: {missing}")
        except Exception as e:
            st.error(f"Could not load pipeline: {e}. Falling back to training.")
            use_saved = False

    if not use_saved:
        # Model training
        st.subheader("Training Models...")
        model_results, trained_models = train_models(
            selected_X_train, X_test_selected, y_train_resampled, y_test,
            cv=int(cv), randomized=bool(randomized), best_only=bool(best_only), use_smote_in_cv=bool(use_smote_in_cv), calibrate=bool(calibrate)
        )
        st.success("✅ Models trained.")

        # Create ensemble model
        ensemble_model, ensemble_accuracy, ensemble_report = create_ensemble(
            X_train=selected_X_train,
            X_test=X_test_selected,
            y_train=y_train_resampled,
            y_test=y_test,
            model_results=model_results,
            models=trained_models
        )
        st.success("✅ Ensemble model created.")

        # Determine best threshold using ensemble probabilities
        try:
            proba_test = ensemble_model.predict_proba(X_test_selected)[:, 1]
            best_t, best_t_score = find_best_threshold(y_test, proba_test, beta=2)
        except Exception:
            best_t, best_t_score = 0.5, None

        # Save full pipeline for deployment
        try:
            if overwrite_after_train:
                save_full_pipeline(preprocessor, selector, ensemble_model, best_t, expected_columns=EXPECTED_COLUMNS, path=pipeline_path)
        except Exception:
            pass

        # Export model results for experiments (cached)
        try:
            cached_export_results(model_results)
            export_model_results(model_results)
        except Exception:
            pass

    # Tabs
    tab_single, tab_batch, tab_history, tab_admin = st.tabs(["Predict Single", "Predict Batch", "History", "Admin/Insights"])

    # Threshold slider (global)
    with tab_single:
        st.subheader("📊 Predict Single Customer Churn")
        threshold = best_t if 'best_t' in locals() else 0.5  # use best threshold if available
        st.info(f"Using decision threshold: {threshold:.2f}")
        st.caption("A lower threshold makes the model more sensitive to potential churners. It will flag customers as 'Churn' even if it has a smaller suspicion (e.g., >25% risk).")

        # Optional: use a single selectbox to pick an existing customer to autofill, with a unique key
        if 'customerID' in df.columns:
            cust_ids = df['customerID'].astype(str).unique().tolist()
            selected_id = st.selectbox("Select existing customerID (optional)", options=["--"] + cust_ids, index=0, key="select_customer")
        else:
            selected_id = "--"

        input_data = {}  # Always initialize input_data here

        raw_cols = [c for c in df.columns if c not in ["Churn"]]
        cols1, cols2 = st.columns(2)
        prefill = None
        if selected_id != "--":
            prefill = df[df['customerID'].astype(str) == selected_id].iloc[0].to_dict()
        
        for i, col in enumerate(raw_cols):
            container = cols1 if i % 2 == 0 else cols2
            with container:
                default_val = prefill[col] if prefill and (col in prefill) else (float(df[col].median()) if pd.api.types.is_numeric_dtype(df[col]) else None)
                if pd.api.types.is_numeric_dtype(df[col]):
                    val = st.number_input(f"{col}", value=float(default_val) if default_val is not None else 0.0, key=f"num_input_{col}")
                else:
                    options = sorted(df[col].dropna().astype(str).unique().tolist())
                    index = 0
                    if default_val is not None and str(default_val) in options:
                        index = options.index(str(default_val))
                    val = st.selectbox(f"{col}", options, index=index if options else 0, key=f"sel_{col}")
                input_data[col] = val

        if st.button("🔍 Predict Churn", key="predict_single_button"):
            try:
                with st.spinner("Making prediction..."):
                    raw_input_df = pd.DataFrame([input_data])
                    transformed_df = transform_new_data(raw_input_df, preprocessor, selector, selected_feature_names)
                    proba = ensemble_model.predict_proba(transformed_df)[:, 1][0]
                    prediction = int(proba >= threshold)
                
                # Determine risk level based on the probability
                if proba > 0.65:
                    risk_level = "High"
                elif proba < 0.35:
                    risk_level = "Low"
                else:
                    risk_level = "Medium"
                
                # Display user-friendly prediction results
                st.success(f"🎯 Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
                st.info(f"Churn Risk: {risk_level}")
                st.info(f"Risk Score: {int(proba*100)}%")
                
                # Retention actions panel
                st.subheader("🎯 Suggested Retention Actions")
                if risk_level == "High":
                    st.error("High Churn Risk: Immediate Action Recommended")
                    st.write("- **Priority Outreach:** Contact the customer within 24 hours to understand their dissatisfaction.")
                    st.write("- **Offer a Significant Discount:** Propose a 1-2 month service credit or a significant discount on their plan.")
                    st.write("- **Service Upgrade:** If they use Fiber Optic, check for service issues. Offer a free speed upgrade or a more stable plan.")
                    st.write("- **Switch Payment Method:** If they use Electronic Check, strongly encourage switching to a credit card or auto-pay for a small bonus.")
                elif risk_level == "Medium":
                    st.warning("Medium Churn Risk: Proactive Engagement Needed")
                    st.write("- **Targeted Offer:** Send a proactive email with a loyalty credit or a small discount on an add-on service.")
                    st.write("- **Promote Contract Benefits:** If they are on a month-to-month contract, highlight the savings of a one or two-year contract.")
                    st.write("- **Value-Added Services:** Offer a free trial of a service they don't have, like Online Security or Device Protection.")
                else: # Low Risk
                    st.success("Low Churn Risk: Focus on Engagement and Upselling")
                    st.write("- **Routine Engagement:** No immediate action needed. Include in standard customer newsletters.")
                    st.write("- **Upsell Opportunity:** Suggest relevant additional services they might benefit from, like Streaming TV or Online Backup.")
                    st.write("- **Referral Program:** Encourage them to refer friends for a mutual benefit, strengthening their loyalty.")
                
                # Explanation fallback: Random Forest SHAP or permutation importance
                try:
                    rf = trained_models.get('Random Forest') if 'trained_models' in locals() else None
                    if rf is not None and hasattr(rf, 'predict_proba'):
                        sample_X = pd.DataFrame(X_test_selected, columns=selected_feature_names).iloc[:100]
                        explainer = shap.TreeExplainer(rf)
                        shap_values = explainer.shap_values(sample_X)
                        st.caption("Feature importance summary (Random Forest)")
                        shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values, sample_X, show=False)
                        st.pyplot(bbox_inches='tight', dpi=120)
                    else:
                        expl_pairs = top_permutation_importances(ensemble_model, selected_X_train.iloc[:200], y_train_resampled.iloc[:200], selected_feature_names, top_k=3)
                        if expl_pairs:
                            st.caption("Top contributing features (permutation importance on sample):")
                            for fname, imp in expl_pairs:
                                st.write(f"- {fname}: {imp:.4f}")
                except Exception:
                    pass
                
                # (Optional: Add any explanation graphs or SHAP summary code below)
                save_prediction_streamlit(input_data, prediction, proba)
                st.success("✅ Prediction saved successfully.")
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")

    # Predict Batch
    with tab_batch:
        st.subheader("📦 Predict Batch (Upload CSV of Customers)")

        st.info(
            "Upload a CSV file with customer data to predict churn for the entire list at once. "
            "The results table below shows a preview of your data with two new columns: 'prediction' and 'proba' (churn probability). "
            "The table is automatically sorted to place the customers with the highest churn risk at the top. "
            "Use the download button to save the complete scored file."
        )
        threshold_b = st.slider("Decision threshold (batch)", min_value=0.1, max_value=0.9, value=float(best_t if 'best_t' in locals() else 0.5), step=0.01)
        batch_file = st.file_uploader("Upload CSV for batch scoring", type=["csv"], key="batch")
        if batch_file is not None:
            try:
                batch_df = pd.read_csv(batch_file)
                batch_df = add_age_range(batch_df)
                if 'TotalCharges' in batch_df.columns:
                    batch_df['TotalCharges'] = pd.to_numeric(batch_df['TotalCharges'], errors='coerce').fillna(0)
                for col in batch_df.columns:
                    if batch_df[col].dtype == object:
                        batch_df[col] = batch_df[col].astype(str).fillna("Unknown")
                transformed_batch = transform_new_data(batch_df, preprocessor, selector, selected_feature_names)
                probas = ensemble_model.predict_proba(transformed_batch)[:, 1]
                preds = (probas >= threshold_b).astype(int)

                out = batch_df.copy()
                out['prediction'] = preds
                out['proba'] = probas
                out = out.sort_values('proba', ascending=False)
                st.dataframe(out.head(50))

                csv = out.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions CSV", csv, "predictions.csv", "text/csv")

                st.success("✅ Batch scoring complete! The preview above shows the top 50 customers with the highest churn risk. Click the download button to save the full results.")

                # Auto-export top risk customers
                try:
                    os.makedirs('ml_results', exist_ok=True)
                    out.to_csv('ml_results/top_risk_customers.csv', index=False)
                except Exception:
                    pass
            except Exception as e:
                st.error(f"❌ Error in batch scoring: {str(e)}")

    # History
    # In app.py, replace your entire "with tab_history:" block with this

    with tab_history:
        st.subheader("🗂️ Prediction History")
        st.info("This table shows a log of all past predictions. Note that the format may vary for older entries.")

        try:
            rows = get_prediction_history()
            if rows:
                # --- DATA LOADING AND CLEANING ---
                cols6 = ["id", "input_data", "prediction", "model_name", "proba", "timestamp"]
                hist_df = pd.DataFrame(rows, columns=cols6)

                # --- ROBUST JSON PARSING LOGIC ---
                def safe_parse(data):
                    try:
                        return json.loads(data)
                    except (json.JSONDecodeError, TypeError):
                        return {}
                parsed_df = pd.DataFrame([safe_parse(x) for x in hist_df['input_data']])
                hist_df = pd.concat([hist_df.drop(columns=['input_data']), parsed_df], axis=1)

                # --- CRITICAL FIX: ENSURE CORRECT DATA TYPES ---
                # Convert 'prediction' to a nullable integer type to handle potential missing values
                if 'prediction' in hist_df.columns:
                    hist_df['prediction'] = pd.to_numeric(hist_df['prediction'], errors='coerce').astype('Int64')

                # --- FILTERS ---
                with st.expander("Filters"):
                    pred_filter = st.selectbox("Prediction filter", ["All", "Churn", "Not Churn"], index=0)
                    search_query = st.text_input("Search text in any column")

                # Apply filters to a copy of the dataframe
                filtered_df = hist_df.copy()

                # Apply prediction filter
                if pred_filter != "All":
                    if 'prediction' in filtered_df.columns:
                        filter_value = 1 if pred_filter == "Churn" else 0
                        # Drop rows where prediction is missing before filtering
                        filtered_df = filtered_df.dropna(subset=['prediction'])
                        filtered_df = filtered_df[filtered_df['prediction'] == filter_value]

                # Apply search filter
                if search_query:
                    # Fill missing values with empty strings for searching
                    df_for_search = filtered_df.fillna('').astype(str)
                    # The search logic is now more reliable
                    mask = df_for_search.apply(lambda row: row.str.contains(search_query, case=False, na=False).any(), axis=1)
                    filtered_df = filtered_df[mask]
                
                # Display the filtered dataframe
                st.dataframe(filtered_df.tail(200))

            else:
                st.info("No predictions saved yet.")
        except Exception as e:
            st.error(f"❌ Could not load history: {str(e)}")

    # Admin/Insights
    with tab_admin:
        st.subheader("🛠️ Admin: Training Results & Insights")
        if 'ensemble_accuracy' in locals() and not np.isnan(ensemble_accuracy):
            st.write(f"Ensemble accuracy: {ensemble_accuracy:.2%}")
            st.caption("This shows the overall accuracy of the final combined model on the test data. It represents the percentage of customers (both churners and non-churners) that the model predicted correctly.")
        try:
            proba_test = ensemble_model.predict_proba(X_test_selected)[:, 1]
            best_t, best_t_score = find_best_threshold(y_test, proba_test, beta=2)
            st.write(f"Best F2 threshold: {best_t:.2f} (F2-Score = {best_t_score:.2f})")
            st.caption("The F2-Score is a metric that prioritizes finding real churners (minimizing false negatives). This is the optimal decision threshold (e.g., 25%) that maximizes this score, making it the best setting for this business problem.")
        except Exception:
            pass

        st.subheader("📈 Model Performance Plots")
        st.info(
            """
            These plots compare the performance of all the models trained during the session.
            - **ROC Curves** show how well each model can distinguish between churners and non-churners (a curve closer to the top-left is better).
            - The **Model Accuracy Comparison** bar chart provides a direct ranking of the models based on their overall accuracy.
            - The **Model Metrics Comparison** line chart shows the trade-off between Precision (how many churn predictions are correct) and Recall (how many actual churners were found).
            """
        )
        try:
            plot_roc_curves(trained_models if 'trained_models' in locals() else {}, X_test_selected, y_test, results_dir="ml_results")
            st.image("ml_results/roc_curves.png")
        except Exception:
            pass

        try:
            if 'model_results' in locals() and model_results:
                plot_results(model_results, results_dir="ml_results")
                col1, col2 = st.columns(2)
                with col1:
                    st.image("ml_results/model_accuracy_comparison.png")
                with col2:
                    st.image("ml_results/model_metrics_comparison.png")
        except Exception:
            pass

        st.subheader("🧩 SHAP Explanations")
        st.info(
            "SHAP (SHapley Additive exPlanations) is a state-of-the-art method for explaining the output of any machine learning model. "
            "The plot below shows a **global summary**, revealing which features have the biggest impact on predicting churn across the entire dataset. "
            "Features at the top are the most influential."
        )
        try:
            # Model selector for SHAP
            choices = []
            if 'trained_models' in locals() and trained_models:
                if 'Random Forest' in trained_models:
                    choices.append('Random Forest')
                if 'XGBoost' in trained_models:
                    choices.append('XGBoost')
            model_choice = st.selectbox("Model for SHAP", options=choices if choices else ["Auto"], index=0)
            target_model = None
            if choices:
                target_model = trained_models.get(model_choice)
            sample_X = pd.DataFrame(X_test_selected, columns=selected_feature_names).iloc[:100]
            if target_model is not None and hasattr(target_model, 'predict_proba'):
                if model_choice == 'Random Forest':
                    explainer = shap.TreeExplainer(target_model)
                    shap_values = explainer.shap_values(sample_X)
                elif model_choice == 'XGBoost':
                    explainer = shap.TreeExplainer(target_model)
                    shap_values = explainer.shap_values(sample_X)
                st.caption(f"Feature importance summary ({model_choice})")
                shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values, sample_X, show=False)
                st.pyplot(bbox_inches='tight', dpi=120)
            else:
                explainer = shap.KernelExplainer(ensemble_model.predict_proba, sample_X[:50])
                shap_values = explainer.shap_values(sample_X[:50], nsamples=100)
                st.caption("Feature importance summary (KernelExplainer, sample)")
                shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values, sample_X[:50], show=False)
                st.pyplot(bbox_inches='tight', dpi=120)
        except Exception:
            st.info("SHAP could not be displayed for the current model.")

        st.subheader("⚖️ Fairness Metrics")
        st.info(
            "This section is crucial for ensuring the model is fair and ethical. It checks if the model's predictions are biased towards any particular demographic group. "
            "The tables below show the **churn rate** and **average churn probability** for different groups (e.g., by gender or senior citizen status). "
            "Ideally, these rates should be roughly similar across all groups, indicating that the model is not unfairly penalizing anyone."
        )
        try:
            # Build a small DataFrame for fairness slices from original df and X_test_selected index alignment may not exist
            # So compute on full df using model via transform
            proba_df = transform_new_data(df.drop(columns=['Churn']), preprocessor, selector, selected_feature_names)
            probs_all = ensemble_model.predict_proba(proba_df)[:, 1]
            preds_all = (probs_all >= best_t).astype(int)
            full = df.copy()
            full['pred'] = preds_all
            full['proba'] = probs_all
            for group_col in ['SeniorCitizen', 'gender']:
                if group_col in full.columns:
                    st.caption(f"Group: {group_col}")
                    grp = full.groupby(group_col).apply(lambda g: pd.Series({
                        'count': len(g),
                        'churn_rate': g['pred'].mean(),
                        'avg_proba': g['proba'].mean(),
                    }))
                    st.dataframe(grp.style.format({
                        'count': '{:,.0f}',
                        'churn_rate': '{:.2%}',
                        'avg_proba': '{:.2%}'
                    }))
        except Exception:
            pass
