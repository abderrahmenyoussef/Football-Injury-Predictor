import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, List

class InjuryEDA:
    def __init__(self):
        plt.style.use('seaborn-v0_8')  # Updated to use correct style name
        self.correlation_features = [
            'Age', 'FIFA rating', 'Injury_Duration',
            'Performance_Impact', 'Form_Before_Injury',
            'Form_After_Return', 'Total_Injuries',
            'Avg_Injury_Duration', 'Overall_Risk_Score'
        ]

    def plot_correlation_heatmap(self, df: pd.DataFrame, figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Generate correlation heatmap for numeric features
        
        Parameters:
        -----------
        df : pd.DataFrame
            The dataset to analyze
        figsize : Tuple[int, int]
            Size of the figure in inches
            
        Returns:
        --------
        plt.Figure
            The matplotlib figure object
        """
        # Select only numeric columns that are relevant for correlation analysis
        numeric_df = df[self.correlation_features].copy()
        
        plt.figure(figsize=figsize)
        corr = numeric_df.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Generate heatmap
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8}
        )
        
        plt.title('Feature Correlation Heatmap', pad=20)
        plt.tight_layout()
        
        return plt.gcf()

    def get_top_correlations(self, df: pd.DataFrame, target_feature: str, top_n: int = 5) -> Dict[str, float]:
        """
        Get the top N features most correlated with a target feature
        
        Parameters:
        -----------
        df : pd.DataFrame
            The dataset to analyze
        target_feature : str
            The feature to find correlations with
        top_n : int
            Number of top correlations to return
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of feature names and their correlation coefficients
        """
        numeric_df = df[self.correlation_features].copy()
        correlations = numeric_df.corr()[target_feature].sort_values(ascending=False)
        
        # Remove self-correlation
        correlations = correlations[correlations.index != target_feature]
        
        return correlations.head(top_n).to_dict()

    def plot_feature_correlations(self, df: pd.DataFrame, target_feature: str) -> plt.Figure:
        """
        Create scatter plots of top correlated features with target feature
        
        Parameters:
        -----------
        df : pd.DataFrame
            The dataset to analyze
        target_feature : str
            The feature to analyze correlations with
            
        Returns:
        --------
        plt.Figure
            The matplotlib figure object
        """
        # Get top 4 correlated features
        top_corr = self.get_top_correlations(df, target_feature, top_n=4)
        
        # Create subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle(f'Top Correlations with {target_feature}', fontsize=16, y=1.02)
        
        for (feature, corr), ax in zip(top_corr.items(), axes.ravel()):
            sns.scatterplot(data=df, x=feature, y=target_feature, ax=ax)
            ax.set_title(f'Correlation: {corr:.2f}')
            
        plt.tight_layout()
        return fig

    def plot_performance_load_analysis(self, df: pd.DataFrame) -> plt.Figure:
        """
        Plot relationship between performance metrics and injury risk
        
        Parameters:
        -----------
        df : pd.DataFrame
            The dataset to analyze
            
        Returns:
        --------
        plt.Figure
            The matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance and Load Analysis', fontsize=16, y=1.02)
        
        # Plot 1: Recent Match Ratings vs Risk Score
        sns.scatterplot(
            data=df,
            x='Avg_Rating_Before_Injury',
            y='Overall_Risk_Score',
            hue='Injury_Severity',
            ax=axes[0, 0]
        )
        axes[0, 0].set_title('Match Ratings vs Injury Risk')
        axes[0, 0].set_xlabel('Average Rating Before Injury')
        axes[0, 0].set_ylabel('Overall Risk Score')
        
        # Plot 2: Form vs Total Previous Injuries
        sns.scatterplot(
            data=df,
            x='Form_Before_Injury',
            y='Total_Injuries',
            hue='Position',
            ax=axes[0, 1]
        )
        axes[0, 1].set_title('Form vs Injury History')
        axes[0, 1].set_xlabel('Form Before Injury')
        axes[0, 1].set_ylabel('Total Previous Injuries')
        
        # Plot 3: Opposition Strength vs Performance
        sns.boxplot(
            data=df,
            x='Injury_Severity',
            y='Recent_Opposition_Strength',
            ax=axes[1, 0]
        )
        axes[1, 0].set_title('Injury Severity by Opposition Strength')
        axes[1, 0].set_xlabel('Injury Severity')
        axes[1, 0].set_ylabel('Opposition Strength')
        
        # Plot 4: Risk Scores Distribution
        risk_data = df[['Age_Risk_Score', 'Position_Risk_Score', 'History_Risk_Score']].melt()
        sns.boxplot(
            data=risk_data,
            x='variable',
            y='value',
            ax=axes[1, 1]
        )
        axes[1, 1].set_title('Risk Factors Distribution')
        axes[1, 1].set_xlabel('Risk Factor')
        axes[1, 1].set_ylabel('Risk Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
        
    def plot_workload_patterns(self, df: pd.DataFrame) -> plt.Figure:
        """
        Analyze patterns in match workload and performance before injuries
        
        Parameters:
        -----------
        df : pd.DataFrame
            The dataset to analyze
            
        Returns:
        --------
        plt.Figure
            The matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Workload and Performance Patterns', fontsize=16, y=1.02)
        
        # Plot 1: Match Ratings Trend
        match_ratings = pd.DataFrame({
            'Match': ['3 Matches Before', '2 Matches Before', '1 Match Before'],
            'Rating': [
                df['Match3_before_injury_Player_rating'].mean(),
                df['Match2_before_injury_Player_rating'].mean(),
                df['Match1_before_injury_Player_rating'].mean()
            ]
        })
        sns.lineplot(
            data=match_ratings,
            x='Match',
            y='Rating',
            marker='o',
            ax=axes[0, 0]
        )
        axes[0, 0].set_title('Average Rating Trend Before Injury')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Performance Impact by Position
        sns.boxplot(
            data=df,
            x='Position',
            y='Performance_Impact',
            ax=axes[0, 1]
        )
        axes[0, 1].set_title('Performance Impact by Position')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Team Performance (GD) Before Injury
        sns.histplot(
            data=df,
            x='Pre_Injury_GD_Trend',
            hue='Injury_Severity',
            multiple="stack",
            ax=axes[1, 0]
        )
        axes[1, 0].set_title('Team Performance Before Injury')
        axes[1, 0].set_xlabel('Goal Difference Trend')
        
        # Plot 4: Match Intensity (Opposition Strength vs Form)
        sns.scatterplot(
            data=df,
            x='Recent_Opposition_Strength',
            y='Form_Before_Injury',
            hue='Injury_Severity',
            size='Total_Injuries',
            sizes=(50, 200),
            alpha=0.6,
            ax=axes[1, 1]
        )
        axes[1, 1].set_title('Match Intensity Analysis')
        axes[1, 1].set_xlabel('Opposition Strength')
        axes[1, 1].set_ylabel('Form')
        
        plt.tight_layout()
        return fig

def main():
    """Example usage of the EDA functions"""
    try:
        # Load the cleaned data
        data_path = Path("data/processed/cleaned_injury_data.csv")
        df = pd.read_csv(data_path)
        
        # Initialize EDA class
        eda = InjuryEDA()
        
        # Generate and save correlation heatmap
        heatmap_fig = eda.plot_correlation_heatmap(df)
        heatmap_fig.savefig('data/visualization/correlation_heatmap.png')
        
        # Get top correlations with injury duration
        top_corr = eda.get_top_correlations(df, 'Injury_Duration')
        print("\nTop correlations with Injury Duration:")
        for feature, corr in top_corr.items():
            print(f"{feature}: {corr:.3f}")
        
        # Generate and save feature correlation plots
        corr_fig = eda.plot_feature_correlations(df, 'Injury_Duration')
        corr_fig.savefig('data/visualization/injury_duration_correlations.png')
        
        # Generate and save performance load analysis
        load_fig = eda.plot_performance_load_analysis(df)
        load_fig.savefig('data/visualization/performance_load_analysis.png')
        
        # Generate and save workload patterns
        workload_fig = eda.plot_workload_patterns(df)
        workload_fig.savefig('data/visualization/workload_patterns.png')
        
        plt.close('all')
        print("\nVisualization files have been saved to the data/visualization directory:")
        print("1. correlation_heatmap.png - Feature correlations")
        print("2. performance_load_analysis.png - Performance metrics vs injury risk")
        print("3. workload_patterns.png - Match workload and performance patterns")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()