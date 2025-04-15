import pandas as pd
import numpy as np
from pathlib import Path

def load_processed_data(dataset_path: str) -> pd.DataFrame:
    """
    Load the processed injury dataset
    
    Parameters:
    -----------
    dataset_path : str
        Path to the processed CSV file
        
    Returns:
    --------
    pd.DataFrame
        Processed injury dataset with all features
    """
    try:
        df = pd.read_csv(dataset_path)
        
        # Validate essential columns are present
        required_columns = [
            'Name', 'Position', 'Age', 'Season', 'FIFA rating',
            'Injury', 'Date of Injury', 'Date of return',
            'Injury_Duration', 'Injury_Severity',
            'Avg_Rating_Before_Injury', 'Avg_Rating_After_Return',
            'Performance_Impact', 'Form_Before_Injury', 'Form_After_Return',
            'Overall_Risk_Score'
        ]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert date columns to datetime
        df['Date of Injury'] = pd.to_datetime(df['Date of Injury'])
        df['Date of return'] = pd.to_datetime(df['Date of return'])
        
        # Validate data types
        assert df['Age'].dtype in ['int64', 'float64'], "Age should be numeric"
        assert df['FIFA rating'].dtype in ['int64', 'float64'], "FIFA rating should be numeric"
        assert df['Injury_Duration'].dtype in ['int64', 'float64'], "Injury duration should be numeric"
        
        return df
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")

def get_injury_statistics(df: pd.DataFrame) -> dict:
    """
    Calculate basic statistics about the injury dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Processed injury dataset
        
    Returns:
    --------
    dict
        Dictionary containing basic statistics about injuries
    """
    stats = {
        'total_injuries': len(df),
        'unique_players': df['Name'].nunique(),
        'injury_types': df['Injury'].value_counts().to_dict(),
        'severity_distribution': df['Injury_Severity'].value_counts().to_dict(),
        'avg_injury_duration': df['Injury_Duration'].mean(),
        'position_distribution': df['Position'].value_counts().to_dict(),
        'avg_performance_impact': df['Performance_Impact'].mean()
    }
    return stats

def main():
    """Example usage of the data loader"""
    try:
        # Adjust the path based on the project structure
        data_path = Path("data/processed/processed_injury_data.csv")
        
        # Load the dataset
        df = load_processed_data(data_path)
        
        # Get basic statistics
        stats = get_injury_statistics(df)
        
        # Print some basic information
        print("\nDataset loaded successfully!")
        print(f"Total number of injuries: {stats['total_injuries']}")
        print(f"Number of unique players: {stats['unique_players']}")
        print(f"Average injury duration: {stats['avg_injury_duration']:.2f} days")
        print(f"Average performance impact: {stats['avg_performance_impact']:.3f}")
        
        return df
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    main()