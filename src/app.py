import streamlit as st
import pandas as pd
import numpy as np
import requests
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from visualization.eda_plots import InjuryEDA
from visualization.risk_dashboard import InjuryRiskDashboard

# Set page configuration
st.set_page_config(
    page_title="Football Injury Predictor",
    page_icon="⚽",
    layout="wide"
)

# API endpoint
API_URL = "http://localhost:5000"

# Load the data
@st.cache_data
def load_data():
    return pd.read_csv('data/processed/cleaned_injury_data.csv')

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_URL}/health")
        return response.status_code == 200
    except:
        return False

def main():
    # Check API status
    api_status = check_api_health()
    if not api_status:
        st.error("⚠️ The API is not running. Please start the Flask API first with 'python src/api.py'")
        st.stop()
    
    # Load data
    df = load_data()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Prediction", "Analytics", "Risk Dashboard"])
    
    if page == "Prediction":
        show_prediction_page()
    elif page == "Analytics":
        show_analytics_page(df)
    else:
        show_risk_dashboard(df)

def show_prediction_page():
    st.title("⚽ Football Injury Predictor")
    st.write("Enter player information to predict injury risk")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=16, max_value=40, value=25)
        fifa_rating = st.number_input("FIFA Rating", min_value=60, max_value=99, value=80)
        form = st.slider("Current Form (0-10)", 0.0, 10.0, 7.0)
        total_injuries = st.number_input("Total Previous Injuries", min_value=0, max_value=20, value=2)
        
    with col2:
        avg_injury_duration = st.number_input("Average Injury Duration (days)", min_value=0, max_value=180, value=30)
        risk_score = st.slider("Overall Risk Score (0-1)", 0.0, 1.0, 0.5)
        recent_opposition = st.number_input("Recent Opposition Strength (0-100)", min_value=0, max_value=100, value=75)
        
    # Prepare input data
    input_data = {
        'Age': age,
        'FIFA rating': fifa_rating,
        'Form_Before_Injury': form,
        'Total_Injuries': total_injuries,
        'Avg_Injury_Duration': avg_injury_duration,
        'Overall_Risk_Score': risk_score,
        'Pre_Injury_GD_Trend': 0.0,
        'Recent_Opposition_Strength': recent_opposition,
        'Age_Risk_Score': age / 40,
        'Position_Risk_Score': 0.5,
        'History_Risk_Score': total_injuries / 10,
        'Form_After_Return': form,
        'Performance_Impact': 0.0,
        'Pre_Injury_GD_Volatility': 0.5,
        'Team_Performance_During_Absence': 0.0
    }
    
    if st.button("Predict Injury Risk"):
        # Make API request for prediction
        try:
            response = requests.post(f"{API_URL}/predict", json=input_data)
            if response.status_code == 200:
                result = response.json()
                
                # Show results
                st.subheader("Prediction Results")
                
                # Display prediction with color coding
                severity = result['prediction']['severity']
                color = {'Minor': 'green', 'Moderate': 'yellow',
                        'Major': 'orange', 'Severe': 'red'}[severity]
                
                st.markdown(f"### Predicted Severity: ::{color}[{severity}]")
                
                # Display probabilities as a bar chart
                st.subheader("Risk Probabilities")
                prob_df = pd.DataFrame({
                    'Severity': list(result['probabilities'].keys()),
                    'Probability': list(result['probabilities'].values())
                })
                st.bar_chart(prob_df.set_index('Severity'))
                
                # Display risk factors
                st.subheader("Risk Factors")
                risk_factors = result['risk_factors']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Age Risk", f"{risk_factors['age_risk']:.2f}")
                with col2:
                    st.metric("History Risk", f"{risk_factors['history_risk']:.2f}")
                with col3:
                    st.metric("Overall Risk", f"{risk_factors['overall_risk']:.2f}")
            else:
                st.error(f"Error from API: {response.json().get('error', 'Unknown error')}")
        except Exception as e:
            st.error(f"Failed to get prediction: {str(e)}")

def show_analytics_page(df):
    st.title("📊 Injury Analytics Dashboard")
    
    # Initialize EDA class
    eda = InjuryEDA()
    
    # Show correlation heatmap
    st.subheader("Feature Correlations")
    fig = eda.plot_correlation_heatmap(df)
    st.pyplot(fig)
    
    # Performance Load Analysis
    st.subheader("Performance and Load Analysis")
    fig = eda.plot_performance_load_analysis(df)
    st.pyplot(fig)
    
    # Workload Patterns
    st.subheader("Workload Patterns")
    fig = eda.plot_workload_patterns(df)
    st.pyplot(fig)

def show_risk_dashboard(df):
    st.title("🎯 Player Risk Dashboard")
    
    # Initialize dashboard
    dashboard = InjuryRiskDashboard()
    
    # Player selection
    players = sorted(df['Name'].unique())
    selected_player = st.selectbox("Select Player", players)
    
    if selected_player:
        player_data = df[df['Name'] == selected_player]
        
        # Show player timeline
        st.subheader("Player Injury Timeline")
        fig = dashboard.plot_player_workload_timeline(df, selected_player)
        st.pyplot(fig)
        
        # Show player stats
        st.subheader("Player Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Injuries", int(player_data['Total_Injuries'].iloc[0]))
        with col2:
            st.metric("Avg. Injury Duration", 
                     f"{player_data['Avg_Injury_Duration'].iloc[0]:.1f} days")
        with col3:
            st.metric("Risk Score", 
                     f"{player_data['Overall_Risk_Score'].iloc[0]:.2f}")

if __name__ == "__main__":
    main()