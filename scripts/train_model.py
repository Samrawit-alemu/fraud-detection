import os
import pandas as pd
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, average_precision_score
from imblearn.over_sampling import SMOTE

# Set up logging for production readiness
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_modeling_pipeline(data_path, model_output_dir="../models"):
    try:
        # 1. Create output directory if it doesn't exist
        os.makedirs(model_output_dir, exist_ok=True)
        
        # 2. Load the dataset
        logging.info("Loading processed dataset...")
        df = pd.read_csv(data_path)
        
        # 3. Separate features and target
        # Make sure to drop non-numeric/raw columns, keeping your engineered features!
        X = df.drop(columns=["class", "signup_time", "purchase_time", "device_id", "source", "browser", "sex", "country"], errors="ignore")
        y = df["class"]
        
        # 4. Stratified Train-Test Split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # 5. Handle Class Imbalance using SMOTE on the training set only
        logging.info("Applying SMOTE to handle class imbalance...")
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        # 6. Initialize and Train the Model
        logging.info("Training Random Forest Classifier...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train_res, y_train_res)
        
        # 7. Evaluate Performance
        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test)[:, 1]
        
        auc_pr = average_precision_score(y_test, y_prob)
        logging.info(f"Model Training Complete! Test AUC-PR: {auc_pr:.4f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        
        # 8. Explicitly Save the Trained Model Artifact
        model_path = os.path.join(model_output_dir, "random_forest_model.pkl")
        joblib.dump(rf, model_path)
        logging.info(f"Model successfully saved and serialized at: {model_path}")
        
        return rf
        
    except FileNotFoundError:
        logging.error(f"Data file not found at {data_path}")
    except Exception as e:
        logging.error(f"An error occurred during the modeling pipeline: {e}")

if __name__ == "__main__":
    # Point this to your engineered dataset
    run_modeling_pipeline("../data/raw/Fraud_Data.csv")