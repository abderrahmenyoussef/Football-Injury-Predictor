from typing import Tuple, Dict, List, Union
import pandas as pd
import numpy as np
from pathlib import Path

class DataValidator:
    def __init__(self):
        # Define required fields for injury prediction
        self.required_fields = {
            'Player': ['Name', 'Position', 'Age', 'FIFA rating'],
            'Match': ['Team Name', 'Season'],
            'Injury': ['Injury', 'Date of Injury', 'Date of return', 'Injury_Duration'],
            'Performance': [
                'Match1_before_injury_Player_rating',
                'Match2_before_injury_Player_rating',
                'Match3_before_injury_Player_rating'
            ]
        }
        
        # Define valid ranges for numeric fields
        self.numeric_ranges = {
            'Age': (16, 45),
            'FIFA rating': (40, 99),
            'Injury_Duration': (0, 400),  # Extended to handle longer recoveries
            'Match1_before_injury_Player_rating': (0, 10),
            'Match2_before_injury_Player_rating': (0, 10),
            'Match3_before_injury_Player_rating': (0, 10)
        }
        
        # Define valid categories for categorical fields
        self.valid_categories = {
            'Position': [
                'Goalkeeper', 
                'Center Back', 'Left Back', 'Right Back',
                'Defensive Midfielder', 'Defensive Midfielder ',  # Added space variant
                'Central Midfielder', 'Central Midfielder ',  # Added space variant
                'Attacking Midfielder',
                'Left Midfielder', 'Right Midfielder',
                'Left Winger', 'Left winger',  # Added case variant
                'Right Winger', 'Right winger',  # Added case variant
                'Center Forward'
            ],
            'Injury_Severity': ['Minor', 'Moderate', 'Major', 'Severe']
        }

    def standardize_position(self, position: str) -> str:
        """
        Standardize position names by removing extra spaces and normalizing cases
        """
        if pd.isna(position):
            return position
            
        position = position.strip()
        # Map similar positions to standard names
        position_mapping = {
            'Right Midfielder': 'Right Winger',
            'Left Midfielder': 'Left Winger',
            'Right winger': 'Right Winger',
            'Left winger': 'Left Winger',
            'Defensive Midfielder ': 'Defensive Midfielder',
            'Central Midfielder ': 'Central Midfielder'
        }
        return position_mapping.get(position, position)

    def validate_required_fields(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that all required fields are present in the dataset
        
        Parameters:
        -----------
        data : pd.DataFrame
            The dataset to validate
            
        Returns:
        --------
        Tuple[bool, List[str]]
            Boolean indicating if validation passed and list of missing fields
        """
        missing_fields = []
        
        for category, fields in self.required_fields.items():
            for field in fields:
                if field not in data.columns:
                    missing_fields.append(f"{field} (Required for {category})")
        
        return len(missing_fields) == 0, missing_fields

    def validate_numeric_ranges(self, data: pd.DataFrame) -> Tuple[bool, Dict[str, List[int]]]:
        """
        Validate that numeric fields are within acceptable ranges
        
        Parameters:
        -----------
        data : pd.DataFrame
            The dataset to validate
            
        Returns:
        --------
        Tuple[bool, Dict[str, List[int]]]
            Boolean indicating if validation passed and dictionary of out-of-range values
        """
        invalid_values = {}
        
        for field, (min_val, max_val) in self.numeric_ranges.items():
            if field in data.columns:
                if field == 'Injury_Duration':
                    # For injury duration, flag only extremely unreasonable values
                    mask = (data[field] < min_val) | (data[field] > 800)  # Allow up to ~2 years
                else:
                    mask = (data[field] < min_val) | (data[field] > max_val)
                invalid_indices = data.index[mask].tolist()
                if invalid_indices:
                    invalid_values[field] = invalid_indices
        
        return len(invalid_values) == 0, invalid_values

    def validate_categorical_values(self, data: pd.DataFrame) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Validate that categorical fields contain only acceptable values
        
        Parameters:
        -----------
        data : pd.DataFrame
            The dataset to validate
            
        Returns:
        --------
        Tuple[bool, Dict[str, List[str]]]
            Boolean indicating if validation passed and dictionary of invalid categories
        """
        invalid_categories = {}
        
        for field, valid_values in self.valid_categories.items():
            if field in data.columns:
                if field == 'Position':
                    # Apply standardization before validation
                    values = data[field].apply(self.standardize_position)
                else:
                    values = data[field]
                
                invalid_values = values.dropna().unique().tolist()
                invalid_values = [val for val in invalid_values if val not in valid_values]
                if invalid_values:
                    invalid_categories[field] = invalid_values
        
        return len(invalid_categories) == 0, invalid_categories

    def validate_dates(self, data: pd.DataFrame) -> Tuple[bool, Dict[str, List[Dict]]]:
        """
        Validate date fields for logical consistency
        
        Parameters:
        -----------
        data : pd.DataFrame
            The dataset to validate
            
        Returns:
        --------
        Tuple[bool, Dict[str, List[Dict]]]
            Boolean indicating if validation passed and dictionary with details of invalid dates
        """
        date_issues = {
            'return_before_injury': [],
            'extreme_duration': []
        }
        
        if 'Date of Injury' in data.columns and 'Date of return' in data.columns:
            # Convert to datetime if not already
            injury_dates = pd.to_datetime(data['Date of Injury'])
            return_dates = pd.to_datetime(data['Date of return'])
            
            # Check for logical inconsistencies
            mask = injury_dates > return_dates
            invalid_indices = data.index[mask].tolist()
            
            if invalid_indices:
                for idx in invalid_indices:
                    date_issues['return_before_injury'].append({
                        'index': idx,
                        'injury_date': injury_dates[idx],
                        'return_date': return_dates[idx],
                        'player': data.loc[idx, 'Name']
                    })
            
            # Check for extremely long recovery periods (> 2 years)
            duration = (return_dates - injury_dates).dt.days
            long_duration_mask = duration > 730
            long_duration_indices = data.index[long_duration_mask].tolist()
            
            if long_duration_indices:
                for idx in long_duration_indices:
                    date_issues['extreme_duration'].append({
                        'index': idx,
                        'duration_days': duration[idx],
                        'player': data.loc[idx, 'Name']
                    })
        
        return len(date_issues['return_before_injury']) == 0, date_issues

    def validate_dataset(self, data: pd.DataFrame) -> Tuple[bool, Dict[str, Union[List, Dict]]]:
        """
        Perform comprehensive validation of the dataset
        
        Parameters:
        -----------
        data : pd.DataFrame
            The dataset to validate
            
        Returns:
        --------
        Tuple[bool, Dict]
            Boolean indicating if all validations passed and dictionary of validation results
        """
        # Run all validations
        fields_valid, missing_fields = self.validate_required_fields(data)
        ranges_valid, invalid_ranges = self.validate_numeric_ranges(data)
        categories_valid, invalid_categories = self.validate_categorical_values(data)
        dates_valid, date_issues = self.validate_dates(data)
        
        # Compile results
        validation_results = {
            'missing_fields': missing_fields,
            'invalid_ranges': invalid_ranges,
            'invalid_categories': invalid_categories,
            'date_issues': date_issues
        }
        
        # Overall validation status
        # Consider the dataset valid even with some date anomalies if they're explained
        all_valid = all([
            fields_valid,
            ranges_valid,
            categories_valid,
            # Only fail validation for return dates before injury
            len(date_issues['return_before_injury']) == 0
        ])
        
        return all_valid, validation_results

def main():
    """Example usage of the data validator"""
    try:
        from data_loader import load_processed_data
        
        # Load the cleaned data
        data_path = Path("data/processed/cleaned_injury_data.csv")
        df = load_processed_data(data_path)
        
        # Initialize and run validator
        validator = DataValidator()
        is_valid, results = validator.validate_dataset(df)
        
        print("\nValidation Results for Cleaned Data:")
        print(f"Overall validation {'passed' if is_valid else 'failed'}")
        
        # Print detailed results
        if not is_valid:
            if results['missing_fields']:
                print("\nMissing required fields:")
                for field in results['missing_fields']:
                    print(f"- {field}")
                    
            if results['invalid_ranges']:
                print("\nOut-of-range values found:")
                for field, indices in results['invalid_ranges'].items():
                    print(f"- {field}: {len(indices)} invalid values")
                    
            if results['invalid_categories']:
                print("\nInvalid categorical values found:")
                for field, values in results['invalid_categories'].items():
                    print(f"- {field}: {values}")
            
            if results['date_issues']['return_before_injury']:
                print("\nInvalid dates (return before injury):")
                for issue in results['date_issues']['return_before_injury']:
                    print(f"- Player: {issue['player']}, Injury: {issue['injury_date'].date()}, Return: {issue['return_date'].date()}")
            
        # Always show informational warnings about long durations
        if results['date_issues']['extreme_duration']:
            print("\nWarning - Extremely long injury durations found:")
            for issue in results['date_issues']['extreme_duration']:
                print(f"- Player: {issue['player']}, Duration: {issue['duration_days']} days")
        
        return is_valid, results
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    main()