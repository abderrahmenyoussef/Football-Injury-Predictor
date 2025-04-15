import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

class InjuryRiskDashboard:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        
    def plot_player_workload_timeline(self, df: pd.DataFrame, player_name: str = None) -> plt.Figure:
        """
        Generate timeline of player workload with injury markers
        
        Parameters:
        -----------
        df : pd.DataFrame
            The injury dataset
        player_name : str, optional
            Name of specific player to analyze. If None, aggregates across all players
            
        Returns:
        --------
        plt.Figure
            The matplotlib figure object
        """
        if player_name:
            data = df[df['Name'] == player_name].copy()
        else:
            data = df.copy()
            
        # Convert dates to datetime
        data['Date of Injury'] = pd.to_datetime(data['Date of Injury'])
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), height_ratios=[2, 1.5, 1.5])
        fig.suptitle(
            f"{'Player: ' + player_name if player_name else 'All Players'} - Workload and Injury Analysis",
            fontsize=16,
            y=0.95
        )
        
        # Plot 1: Performance Ratings Timeline
        self._plot_performance_timeline(data, axes[0])
        
        # Plot 2: Risk Score Evolution
        self._plot_risk_evolution(data, axes[1])
        
        # Plot 3: Injury Distribution
        self._plot_injury_distribution(data, axes[2])
        
        plt.tight_layout()
        return fig
    
    def _plot_performance_timeline(self, data: pd.DataFrame, ax: plt.Axes):
        """Plot performance metrics timeline"""
        # Create timeline of match ratings
        timeline_data = []
        
        for _, row in data.iterrows():
            injury_date = pd.to_datetime(row['Date of Injury'])
            return_date = pd.to_datetime(row['Date of return'])
            
            # Add pre-injury matches
            for i in range(1, 4):
                timeline_data.append({
                    'date': injury_date - timedelta(days=i*7),
                    'rating': row[f'Match{i}_before_injury_Player_rating'],
                    'period': 'Pre-Injury',
                    'injury_id': len(timeline_data)
                })
            
            # Add post-injury matches
            for i in range(1, 4):
                timeline_data.append({
                    'date': return_date + timedelta(days=i*7),
                    'rating': row[f'Match{i}_after_injury_Player_rating'],
                    'period': 'Post-Return',
                    'injury_id': len(timeline_data)
                })
        
        timeline_df = pd.DataFrame(timeline_data)
        
        # Plot performance timeline
        if len(timeline_df) > 0:  # Only plot if we have data
            sns.scatterplot(
                data=timeline_df,
                x='date',
                y='rating',
                hue='period',
                style='period',
                ax=ax
            )
            
            # Mark injuries with vertical lines
            for injury_date in data['Date of Injury'].unique():
                ax.axvline(x=pd.to_datetime(injury_date), color='red', 
                          linestyle='--', alpha=0.5)
        
        ax.set_title('Performance Rating Timeline')
        ax.set_xlabel('Date')
        ax.set_ylabel('Match Rating')
        ax.tick_params(axis='x', rotation=45)
        
    def _plot_risk_evolution(self, data: pd.DataFrame, ax: plt.Axes):
        """Plot risk score evolution"""
        # Sort data by date
        data = data.sort_values('Date of Injury')
        
        # Plot risk scores
        ax.plot(data['Date of Injury'], data['Overall_Risk_Score'],
                marker='o', label='Overall Risk')
        ax.plot(data['Date of Injury'], data['History_Risk_Score'],
                marker='s', label='History Risk')
        ax.plot(data['Date of Injury'], data['Age_Risk_Score'],
                marker='^', label='Age Risk')
        
        ax.set_title('Risk Score Evolution')
        ax.set_xlabel('Date')
        ax.set_ylabel('Risk Score')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        
    def _plot_injury_distribution(self, data: pd.DataFrame, ax: plt.Axes):
        """Plot injury distribution over time"""
        # Create monthly injury counts
        data['Month'] = data['Date of Injury'].dt.to_period('M')
        monthly_injuries = data.groupby('Month').size().reset_index()
        monthly_injuries['Month'] = monthly_injuries['Month'].astype(str)
        
        # Plot injury distribution
        sns.barplot(
            data=monthly_injuries,
            x='Month',
            y=0,
            ax=ax
        )
        
        ax.set_title('Monthly Injury Distribution')
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Injuries')
        ax.tick_params(axis='x', rotation=45)

    def create_player_dashboard(self, df: pd.DataFrame, player_name: str) -> Dict[str, plt.Figure]:
        """
        Create a complete dashboard for a specific player
        
        Parameters:
        -----------
        df : pd.DataFrame
            The injury dataset
        player_name : str
            Name of the player to analyze
            
        Returns:
        --------
        Dict[str, plt.Figure]
            Dictionary containing all dashboard figures
        """
        player_data = df[df['Name'] == player_name].copy()
        
        if len(player_data) == 0:
            raise ValueError(f"No data found for player: {player_name}")
        
        # Generate timeline visualization
        timeline_fig = self.plot_player_workload_timeline(df, player_name)
        
        return {
            'timeline': timeline_fig
        }

def main():
    """Example usage of the risk dashboard"""
    try:
        # Load the cleaned data
        data_path = Path("data/processed/cleaned_injury_data.csv")
        df = pd.read_csv(data_path)
        
        # Initialize dashboard
        dashboard = InjuryRiskDashboard()
        
        # Generate overall timeline
        overall_fig = dashboard.plot_player_workload_timeline(df)
        overall_fig.savefig('data/visualization/overall_workload_timeline.png')
        
        # Generate player-specific dashboard
        # Choose a player with multiple injuries for demonstration
        sample_player = df.groupby('Name')['Injury'].count().sort_values(ascending=False).index[0]
        player_dashboard = dashboard.create_player_dashboard(df, sample_player)
        player_dashboard['timeline'].savefig(f'data/visualization/player_workload_timeline_{sample_player}.png')
        
        plt.close('all')
        print("\nVisualization files have been saved:")
        print("1. overall_workload_timeline.png - Aggregated workload analysis")
        print(f"2. player_workload_timeline_{sample_player}.png - Individual player dashboard")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()