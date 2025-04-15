import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict

class InjuryDataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.categorical_columns = [
            'Position', 
            'Injury',
            'Injury_Severity',
            'Team Name'
        ]
        self.numeric_columns = None  # Will be detected automatically

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean missing values and format data
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw injury dataset
            
        Returns:
        --------
        pd.DataFrame
            Cleaned dataset
        """
        df = df.copy()
        
        # Handle missing values in numeric columns
        if self.numeric_columns:
            for col in self.numeric_columns:
                if col in df.columns:
                    # Fill missing values with median for numeric columns
                    df[col] = df[col].fillna(df[col].median())
        
        # Handle missing values in categorical columns
        for col in self.categorical_columns:
            if col in df.columns:
                # Fill missing values with mode for categorical columns
                df[col] = df[col].fillna(df[col].mode()[0])
        
        # Clean and standardize categorical values
        if 'Position' in df.columns:
            df['Position'] = df['Position'].str.strip()
            
        if 'Team Name' in df.columns:
            df['Team Name'] = df['Team Name'].str.strip()
            
        if 'Injury' in df.columns:
            df['Injury'] = df['Injury'].str.strip()
        
        return df

    def encode_categorical_variables(self, df: pd.DataFrame, mode: str = 'train') -> pd.DataFrame:
        """
        Encode categorical variables using Label Encoding and One-Hot Encoding
        
        Parameters:
        -----------
        df : pd.DataFrame
            Cleaned dataset
        mode : str
            'train' for fitting encoders, 'test' for using existing encoders
            
        Returns:
        --------
        pd.DataFrame
            Encoded dataset
        """
        df = df.copy()
        
        # Label encoding for ordinal categories
        ordinal_categories = ['Injury_Severity']
        for col in ordinal_categories:
            if col in df.columns:
                if mode == 'train':
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col])
                else:
                    df[col] = self.label_encoders[col].transform(df[col])
        
        # One-hot encoding for nominal categories
        nominal_categories = ['Position', 'Injury', 'Team Name']
        df = pd.get_dummies(df, columns=[col for col in nominal_categories if col in df.columns])
        
        return df

    def scale_numeric_features(self, df: pd.DataFrame, mode: str = 'train') -> Tuple[pd.DataFrame, StandardScaler]:
        """
        Scale numeric features using StandardScaler
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with encoded categorical variables
        mode : str
            'train' for fitting scaler, 'test' for using existing scaler
            
        Returns:
        --------
        Tuple[pd.DataFrame, StandardScaler]
            Scaled dataset and the fitted scaler
        """
        df = df.copy()
        
        # Automatically detect numeric columns
        if self.numeric_columns is None or mode == 'train':
            self.numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            # Remove any one-hot encoded columns
            self.numeric_columns = [col for col in self.numeric_columns 
                                 if not any(col.startswith(prefix + '_') 
                                          for prefix in self.categorical_columns)]
        
        if self.numeric_columns:
            if mode == 'train':
                df[self.numeric_columns] = self.scaler.fit_transform(df[self.numeric_columns])
            else:
                df[self.numeric_columns] = self.scaler.transform(df[self.numeric_columns])
        
        return df, self.scaler

    def prepare_data(self, df: pd.DataFrame, mode: str = 'train') -> Tuple[pd.DataFrame, Dict]:
        """
        Complete data preprocessing pipeline
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw dataset
        mode : str
            'train' for fitting preprocessor, 'test' for using existing preprocessor
            
        Returns:
        --------
        Tuple[pd.DataFrame, Dict]
            Fully preprocessed dataset and preprocessing artifacts (encoders, scaler)
        """
        # Clean the data
        df = self.clean_data(df)
        
        # Encode categorical variables
        df = self.encode_categorical_variables(df, mode)
        
        # Scale numeric features
        df, scaler = self.scale_numeric_features(df, mode)
        
        # Return preprocessed data and preprocessing artifacts
        artifacts = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'numeric_columns': self.numeric_columns,
            'categorical_columns': self.categorical_columns
        }
        
        return df, artifacts

def main():
    """Example usage of the preprocessor"""
    try:
        from data_loader import load_processed_data
        from pathlib import Path
        
        # Load the data
        data_path = Path("data/processed/processed_injury_data.csv")
        df = load_processed_data(data_path)
        
        # Initialize and run preprocessor
        preprocessor = InjuryDataPreprocessor()
        processed_df, artifacts = preprocessor.prepare_data(df, mode='train')
        
        print("\nPreprocessing completed successfully!")
        print(f"Processed dataset shape: {processed_df.shape}")
        print("\nNumeric columns after scaling:")
        for col in artifacts['numeric_columns']:
            if col in processed_df.columns:
                print(f"{col}: mean={processed_df[col].mean():.3f}, std={processed_df[col].std():.3f}")
        
        return processed_df
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    main()