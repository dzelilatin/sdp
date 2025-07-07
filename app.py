from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from ml_analysis import create_results_directory, load_and_preprocess_data, prepare_features, train_models, create_ensemble, plot_results, save_results

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'ml_results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle dataset upload."""
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and file.filename.endswith('.csv'):
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        return redirect(url_for('train_model', file_path=file_path))
    else:
        return "Invalid file type. Please upload a CSV file."

@app.route('/train')
def train_model():
    """Train models and display results."""
    file_path = request.args.get('file_path')
    try:
        # Load and preprocess data
        df = load_and_preprocess_data(file_path)
        X_train, X_test, y_train, y_test = prepare_features(df)

        # Train models
        model_results = train_models(X_train, X_test, y_train, y_test)

        # Initialize models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42),
            'SVM': SVC(probability=True)
        }

        # Create ensemble
        ensemble_accuracy, ensemble_report = create_ensemble(
            X_train, X_test, y_train, y_test, model_results, models
        )

        # Create visualization
        plot_results(model_results, RESULTS_FOLDER)

        # Save results
        save_results(model_results, ensemble_accuracy, ensemble_report, RESULTS_FOLDER)

        return render_template('results.html', model_results=model_results, ensemble_accuracy=ensemble_accuracy)

    except Exception as e:
        return render_template('error.html', error_message=str(e))

@app.route('/results/<filename>')
def download_results(filename):
    """Download results or plots."""
    return send_from_directory(RESULTS_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)