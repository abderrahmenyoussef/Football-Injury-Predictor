import pandas as pd
import numpy as np
from datetime import datetime
import re

class InjuryDataProcessor:
    def __init__(self):
        self.injuries_df = None
        self.players_df = None
        self.combined_df = None

    def load_data(self):
        """Load both datasets"""
        self.injuries_df = pd.read_csv('data/player_injuries_impact.csv')
        
        # Load and parse the player data file
        with open('data/Final-player.txt', 'r') as f:
            player_data = f.read()
        
        # Convert string representation to actual Python data
        # Remove 'u' prefixes and normalize quotes
        player_data = player_data.replace("u'", "'").replace('\xa0', ' ')
        try:
            player_list = eval(player_data)
        except:
            print("Error parsing player data. Using simplified loading...")
            # Create a minimal DataFrame that can still merge with injuries data
            self.players_df = pd.DataFrame(columns=['Name'])
            return
            
        # Convert to DataFrame with basic fields we're confident about
        players = []
        for player in player_list:
            try:
                # Only extract fields we're confident about and need
                player_dict = {
                    'Name': str(player[1]).strip() if len(player) > 1 else '',
                    'Club': str(player[2]).strip() if len(player) > 2 else '',
                }
                players.append(player_dict)
            except Exception as e:
                print(f"Error processing player: {e}")
                continue
            
        self.players_df = pd.DataFrame(players)
        
        # Basic cleaning
        self.players_df['Name'] = self.players_df['Name'].str.strip()
        
    def clean_dates(self):
        """Clean and convert date columns to datetime"""
        date_columns = ['Date of Injury', 'Date of return']
        
        def parse_date(date_str):
            if pd.isna(date_str):
                return pd.NaT
            date_str = date_str.strip()
            try:
                for fmt in ['%b %d, %Y', '%b %d,%Y', '%B %d, %Y', '%B %d,%Y']:
                    try:
                        return pd.to_datetime(date_str, format=fmt)
                    except:
                        continue
                return pd.to_datetime(date_str)
            except:
                return pd.NaT
            
        for col in date_columns:
            self.injuries_df[col] = self.injuries_df[col].apply(parse_date)

    def clean_ratings(self, rating):
        """Clean player rating values"""
        if pd.isna(rating) or rating == 'N.A.':
            return np.nan
        # Extract first number if it contains (S) for substitute
        if '(S)' in str(rating):
            rating = str(rating).replace('(S)', '')
        try:
            return float(rating)
        except:
            return np.nan
            
    def calculate_injury_duration(self):
        """Calculate injury duration in days"""
        self.injuries_df['Injury_Duration'] = (
            self.injuries_df['Date of return'] - self.injuries_df['Date of Injury']
        ).dt.days
        
        # Add severity classification
        self.injuries_df['Injury_Severity'] = pd.cut(
            self.injuries_df['Injury_Duration'],
            bins=[0, 7, 21, 60, float('inf')],
            labels=['Minor', 'Moderate', 'Major', 'Severe']
        )
        
    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        # Pre-injury performance
        rating_cols_before = [
            'Match1_before_injury_Player_rating',
            'Match2_before_injury_Player_rating',
            'Match3_before_injury_Player_rating'
        ]
        
        # Post-injury performance
        rating_cols_after = [
            'Match1_after_injury_Player_rating',
            'Match2_after_injury_Player_rating',
            'Match3_after_injury_Player_rating'
        ]
        
        # Clean all ratings
        for col in rating_cols_before + rating_cols_after:
            self.injuries_df[col] = self.injuries_df[col].apply(self.clean_ratings)
            
        # Calculate average ratings
        self.injuries_df['Avg_Rating_Before_Injury'] = self.injuries_df[rating_cols_before].mean(axis=1)
        self.injuries_df['Avg_Rating_After_Return'] = self.injuries_df[rating_cols_after].mean(axis=1)
        
        # Calculate performance impact
        self.injuries_df['Performance_Impact'] = (
            self.injuries_df['Avg_Rating_After_Return'] - self.injuries_df['Avg_Rating_Before_Injury']
        )
        
        # Calculate form and results
        result_mapping = {'win': 3, 'draw': 1, 'lose': 0}
        
        def calculate_form(df, cols):
            form = 0
            valid_results = 0
            for col in cols:
                result = str(df[col]).lower()
                if result in result_mapping:
                    form += result_mapping[result]
                    valid_results += 1
            return form / valid_results if valid_results > 0 else np.nan
        
        result_cols_before = [
            'Match1_before_injury_Result',
            'Match2_before_injury_Result',
            'Match3_before_injury_Result'
        ]
        
        result_cols_after = [
            'Match1_after_injury_Result',
            'Match2_after_injury_Result',
            'Match3_after_injury_Result'
        ]
        
        self.injuries_df['Form_Before_Injury'] = self.injuries_df.apply(
            lambda x: calculate_form(x, result_cols_before), axis=1
        )
        
        self.injuries_df['Form_After_Return'] = self.injuries_df.apply(
            lambda x: calculate_form(x, result_cols_after), axis=1
        )
        
        # Calculate form impact
        self.injuries_df['Form_Impact'] = self.injuries_df['Form_After_Return'] - self.injuries_df['Form_Before_Injury']

    def calculate_injury_history(self):
        """Calculate detailed injury history features"""
        injury_history = self.injuries_df.groupby('Name').agg({
            'Injury': ['count', lambda x: x.nunique()],  # Total injuries and unique types
            'Injury_Duration': ['mean', 'sum', 'std'],  # Duration statistics
            'Injury_Severity': lambda x: (x == 'Severe').sum()  # Count of severe injuries
        }).reset_index()
        
        injury_history.columns = [
            'Name', 'Total_Injuries', 'Unique_Injury_Types',
            'Avg_Injury_Duration', 'Total_Days_Injured',
            'Injury_Duration_Std', 'Severe_Injuries'
        ]
        
        # Calculate injury frequency
        injury_history['Injury_Frequency'] = injury_history['Total_Injuries'] / (
            injury_history['Total_Days_Injured'] + 1
        ) * 365  # Injuries per year
        
        self.injuries_df = pd.merge(self.injuries_df, injury_history, on='Name', how='left')
        
    def calculate_match_load(self):
        """Calculate detailed match load features"""
        gd_cols_before = [
            'Match1_before_injury_GD',
            'Match2_before_injury_GD',
            'Match3_before_injury_GD'
        ]
        
        gd_cols_missed = [
            'Match1_missed_match_GD',
            'Match2_missed_match_GD',
            'Match3_missed_match_GD'
        ]
        
        # Clean GD values
        for col in gd_cols_before + gd_cols_missed:
            self.injuries_df[col] = pd.to_numeric(self.injuries_df[col], errors='coerce')
            
        # Calculate GD trends and volatility
        self.injuries_df['Pre_Injury_GD_Trend'] = self.injuries_df[gd_cols_before].mean(axis=1)
        self.injuries_df['Pre_Injury_GD_Volatility'] = self.injuries_df[gd_cols_before].std(axis=1)
        
        # Team performance during absence
        self.injuries_df['Team_Performance_During_Absence'] = self.injuries_df[gd_cols_missed].mean(axis=1)
        
        # Calculate match intensity
        opposition_strength = {
            'Man City': 5, 'Liverpool': 5, 'Chelsea': 4, 'Arsenal': 4,
            'Tottenham': 4, 'Man United': 4, 'Leicester': 3, 'West Ham': 3,
            'Wolves': 3, 'Brighton': 3, 'Newcastle': 3, 'Crystal Palace': 2,
            'Brentford': 2, 'Aston Villa': 2, 'Southampton': 2, 'Leeds': 2,
            'Everton': 2, 'Nottm Forest': 1, 'Burnley': 1, 'Sheffield': 1
        }
        
        def get_opposition_strength(opposition):
            if pd.isna(opposition):
                return np.nan
            return opposition_strength.get(opposition, 2)  # Default to 2 for unknown teams
        
        self.injuries_df['Recent_Opposition_Strength'] = self.injuries_df['Match1_before_injury_Opposition'].apply(
            get_opposition_strength
        )
        
    def calculate_risk_factors(self):
        """Calculate comprehensive risk factors"""
        # Age-based risk factors
        self.injuries_df['Age_Risk_Score'] = self.injuries_df['Age'].apply(
            lambda x: min((x - 20) / 15, 1) if x >= 20 else 0.1
        )
        
        # Position-based risk factors
        position_risk = {
            'Center Back': 0.8,
            'Left Back': 0.7,
            'Right Back': 0.7,
            'Defensive Midfielder': 0.6,
            'Central Midfielder': 0.5,
            'Attacking Midfielder': 0.4,
            'Left winger': 0.5,
            'Right winger': 0.5,
            'Center Forward': 0.6,
            'Goalkeeper': 0.2
        }
        self.injuries_df['Position_Risk_Score'] = self.injuries_df['Position'].map(position_risk)
        
        # Injury history risk
        self.injuries_df['History_Risk_Score'] = (
            (self.injuries_df['Total_Injuries'] * 0.4) +
            (self.injuries_df['Severe_Injuries'] * 0.3) +
            (self.injuries_df['Injury_Frequency'] * 0.3)
        ) / 10  # Normalize to 0-1 range
        
        # Calculate overall risk score
        self.injuries_df['Overall_Risk_Score'] = (
            self.injuries_df['Age_Risk_Score'] * 0.25 +
            self.injuries_df['Position_Risk_Score'] * 0.25 +
            self.injuries_df['History_Risk_Score'] * 0.5
        )
        
    def merge_player_data(self):
        """Merge player profile data with injury data"""
        self.combined_df = pd.merge(
            self.injuries_df,
            self.players_df,
            on=['Name'],
            how='left'
        )
        
    def create_features(self):
        """Create final feature set for modeling"""
        self.clean_dates()
        self.calculate_injury_duration()
        self.calculate_performance_metrics()
        self.calculate_injury_history()
        self.calculate_match_load()
        self.calculate_risk_factors()
        self.merge_player_data()
        
    def get_processed_data(self):
        """Return the processed dataset"""
        if self.combined_df is None:
            print("Please run create_features() first")
            return None
        return self.combined_df

def main():
    processor = InjuryDataProcessor()
    processor.load_data()
    processor.create_features()
    processed_data = processor.get_processed_data()
    
    # Save processed data in the processed subfolder
    output_path = 'data/processed/processed_injury_data.csv'
    processed_data.to_csv(output_path, index=False)
    print(f"Data processing completed. Processed data saved to '{output_path}'")

if __name__ == "__main__":
    main()