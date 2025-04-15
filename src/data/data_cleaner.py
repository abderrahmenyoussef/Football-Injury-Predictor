import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np

def fix_incorrect_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix cases where return dates are before injury dates by swapping them
    """
    df = df.copy()
    
    # Convert dates to datetime if they aren't already
    df['Date of Injury'] = pd.to_datetime(df['Date of Injury'])
    df['Date of return'] = pd.to_datetime(df['Date of return'])
    
    # Find cases where return date is before injury date
    mask = df['Date of Injury'] > df['Date of return']
    
    # Swap dates for these cases
    if mask.any():
        temp = df.loc[mask, 'Date of Injury'].copy()
        df.loc[mask, 'Date of Injury'] = df.loc[mask, 'Date of return']
        df.loc[mask, 'Date of return'] = temp
        
        # Recalculate injury duration
        df['Injury_Duration'] = (df['Date of return'] - df['Date of Injury']).dt.days
        
        print(f"Fixed {mask.sum()} cases of incorrect date ordering")
    
    return df

def fix_extreme_durations(df: pd.DataFrame, threshold_days: int = 800) -> pd.DataFrame:
    """
    Handle cases with extremely long injury durations
    """
    df = df.copy()
    
    # Find cases with extreme durations
    mask = df['Injury_Duration'] > threshold_days
    
    if mask.any():
        # For these cases, we'll cap the duration and adjust the return date
        df.loc[mask, 'Injury_Duration'] = threshold_days
        df.loc[mask, 'Date of return'] = df.loc[mask, 'Date of Injury'] + pd.Timedelta(days=threshold_days)
        
        print(f"Capped {mask.sum()} cases of extreme injury durations to {threshold_days} days")
    
    return df

def standardize_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize position names
    """
    df = df.copy()
    
    position_mapping = {
        'Right Midfielder': 'Right Winger',
        'Left Midfielder': 'Left Winger',
        'Right winger': 'Right Winger',
        'Left winger': 'Left Winger',
        'Defensive Midfielder ': 'Defensive Midfielder',
        'Central Midfielder ': 'Central Midfielder'
    }
    
    # Clean up position names
    df['Position'] = df['Position'].str.strip()
    df['Position'] = df['Position'].replace(position_mapping)
    
    return df

def clean_dataset(input_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Apply all cleaning operations to the dataset
    
    Parameters:
    -----------
    input_path : Path
        Path to the input CSV file
    output_path : Path
        Path to save the cleaned CSV file
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataset
    """
    # Load the data
    df = pd.read_csv(input_path)
    
    # Apply cleaning operations
    df = fix_incorrect_dates(df)
    df = fix_extreme_durations(df)
    df = standardize_positions(df)
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to: {output_path}")
    
    return df

def main():
    """Run the data cleaning process"""
    try:
        input_path = Path("data/processed/processed_injury_data.csv")
        output_path = Path("data/processed/cleaned_injury_data.csv")
        
        # Clean the data
        cleaned_df = clean_dataset(input_path, output_path)
        
        # Print summary statistics
        print("\nDataset cleaning summary:")
        print(f"Total records: {len(cleaned_df)}")
        print(f"Unique players: {cleaned_df['Name'].nunique()}")
        print(f"Average injury duration: {cleaned_df['Injury_Duration'].mean():.1f} days")
        print(f"Position distribution:\n{cleaned_df['Position'].value_counts()}")
        
        return cleaned_df
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    main()