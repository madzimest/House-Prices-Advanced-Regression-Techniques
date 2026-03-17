import sys
from pathlib import Path

# Add the DS folder to sys.path so that 'src' module is found
# app.py is in DS/app/, so parent directory is DS/
sys.path.append(str(Path(__file__).parent.parent))

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the saved stacking pipeline
pipeline = joblib.load('../deployment/stacking_pipeline.pkl')
fold_models = pipeline['fold_models']
meta_model = pipeline['meta_model']
base_model_names = pipeline['base_model_names']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])

        # Generate predictions from each fold
        fold_preds = {name: np.zeros(len(input_df)) for name in base_model_names}
        for fold_dict in fold_models:
            ft = fold_dict['ft']
            ohe = fold_dict['ohe']
            num_cols = fold_dict['num_cols']
            cat_cols = fold_dict['cat_cols']

            X_t = ft.transform(input_df)
            X_cat = ohe.transform(X_t[cat_cols])
            X_num = X_t[num_cols].values
            X_proc = np.hstack([X_num, X_cat])

            for name in base_model_names:
                model = fold_dict[name]
                fold_preds[name] += model.predict(X_proc)

        # Average across folds
        for name in base_model_names:
            fold_preds[name] /= len(fold_models)

        # Stack and apply meta-model
        X_meta = np.column_stack([fold_preds[name] for name in base_model_names])
        log_pred = meta_model.predict(X_meta)[0]
        price = np.expm1(log_pred)

        return jsonify({'predicted_price': round(price, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
