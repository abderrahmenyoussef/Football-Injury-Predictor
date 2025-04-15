import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import joblib

class InjuryModelTrainer:
    def __init__(self):
        # Define model parameters for grid search
        self.model_params = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'class_weight': ['balanced']
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=42),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'class_weight': ['balanced'],
                    'max_iter': [1000]
                }
            }
        }
        self.best_model = None
        self.best_score = 0
        self.scaler = StandardScaler()
        
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """
        Prepare features and target variable with improved feature engineering
        """
        df = df.copy()
        
        # Enhanced feature set
        feature_columns = [
            'Age', 'FIFA rating',
            'Form_Before_Injury', 'Total_Injuries',
            'Avg_Injury_Duration', 'Overall_Risk_Score',
            'Pre_Injury_GD_Trend', 'Recent_Opposition_Strength',
            'Age_Risk_Score', 'Position_Risk_Score', 'History_Risk_Score',
            'Form_After_Return', 'Performance_Impact',
            'Pre_Injury_GD_Volatility', 'Team_Performance_During_Absence'
        ]
        
        # Handle missing values in features
        for col in feature_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Create target variable (injury severity as classification task)
        severity_mapping = {
            'Minor': 0,
            'Moderate': 1,
            'Major': 2,
            'Severe': 3
        }
        
        # Drop rows where Injury_Severity is missing
        df = df.dropna(subset=['Injury_Severity'])
        
        X = df[feature_columns]
        y = df['Injury_Severity'].map(severity_mapping)
        
        return X, y
        
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train multiple models using cross-validation and grid search
        """
        print("\nTraining models with cross-validation and grid search...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        best_models = {}
        
        # Train and tune each model
        for name, model_info in self.model_params.items():
            print(f"\nTraining {name}...")
            
            # Create grid search with cross-validation
            grid_search = GridSearchCV(
                model_info['model'],
                model_info['params'],
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1
            )
            
            # Fit model
            grid_search.fit(X_train_scaled, y_train)
            
            # Store best model and score
            best_models[name] = {
                'model': grid_search.best_estimator_,
                'score': grid_search.best_score_,
                'params': grid_search.best_params_
            }
            
            print(f"{name} best CV score: {grid_search.best_score_:.4f}")
            print(f"Best parameters: {grid_search.best_params_}")
            
            # Update best overall model
            if grid_search.best_score_ > self.best_score:
                self.best_score = grid_search.best_score_
                self.best_model = {
                    'name': name,
                    'model': grid_search.best_estimator_
                }
                
        print(f"\nBest model: {self.best_model['name']} (CV score: {self.best_score:.4f})")
        
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Evaluate the best model on test data with detailed metrics
        """
        print("\nEvaluating best model on test data...")
        
        # Scale test data
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get predictions
        best_model = self.best_model['model']
        y_pred = best_model.predict(X_test_scaled)
        
        # Print evaluation metrics
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
    def save_models(self, output_dir: Path):
        """
        Save trained models and scaler with metadata
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save best model
        model_path = output_dir / f"{self.best_model['name']}_model.pkl"
        joblib.dump(self.best_model['model'], model_path)
        
        # Save scaler
        scaler_path = output_dir / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        
        # Save model metadata
        metadata = {
            'best_model': self.best_model['name'],
            'best_score': self.best_score,
            'feature_columns': self.prepare_features.__doc__
        }
        metadata_path = output_dir / "model_metadata.json"
        pd.Series(metadata).to_json(metadata_path)
        
        print(f"\nModels and metadata saved to {output_dir}")
        
def main():
    """Train and save injury prediction models"""
    try:
        # Load cleaned data
        data_path = Path("data/processed/cleaned_injury_data.csv")
        df = pd.read_csv(data_path)
        
        # Initialize trainer
        trainer = InjuryModelTrainer()
        
        # Prepare features and target
        X, y = trainer.prepare_features(df)
        
        # Split data with more balanced ratio (70/30)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.3,  # 70% for training
            random_state=42,
            stratify=y  # Maintain class distribution
        )
        
        print("\nData split summary:")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Print class distribution
        print("\nClass distribution:")
        print(pd.Series(y_train).value_counts(normalize=True))
        
        # Train models
        trainer.train_models(X_train, y_train)
        
        # Evaluate on test set
        trainer.evaluate_model(X_test, y_test)
        
        # Save models
        output_dir = Path("models/trained")
        trainer.save_models(output_dir)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()