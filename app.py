import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import silhouette_score
import warnings
from datetime import datetime
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Investment Customer Segmentation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #003087;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: black;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 6px solid #003087;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .cluster-description {
        background-color: black;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #c2d6f0;
        margin: 1rem 0;
    }
    .recommendation-box {
        background-color: black;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 6px solid #28a745;
        margin: 1rem 0;
    }
    .stButton > button {
        background-color: #003087;
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        border: none;
        font-size: 0.85rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #0052cc;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
    }
    .welcome-box {
        background: linear-gradient(135deg, #003087 0%, #1f77b4 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stat-card {
        background-color: black;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class InvestmentSegmentationApp:
    """
    Investment Customer Segmentation Streamlit Application
    """
    
    def __init__(self):
        self.models_loaded = False
        self.best_model = 'kmeans'
        self.silhouette_scores = {
            'kmeans': None,
            'gmm': None,
            'dbscan': None
        }
        self.load_models_and_data()
        
    def load_models_and_data(self):
        """Load trained models and preprocessors"""
        try:
            self.scaler = joblib.load("models/scaler.pkl")
            self.kmeans_model = joblib.load("models/kmeans_model.pkl")
            self.gmm_model = joblib.load("models/gmm_model.pkl")
            self.dbscan_model = joblib.load("models/dbscan_model.pkl")
            self.label_encoders = joblib.load("models/label_encoders.pkl")
            self.pca = joblib.load("models/pca_model.pkl")
            
            if os.path.exists("models/model_metadata.json"):
                with open("models/model_metadata.json", 'r') as f:
                    self.metadata = json.load(f)
                self.silhouette_scores = self.metadata.get('silhouette_scores', {})
            else:
                self.metadata = {}
            
            if os.path.exists("outputs/clustered_dataset.csv"):
                self.df_clustered = pd.read_csv("outputs/clustered_dataset.csv")
            else:
                self.df_clustered = pd.DataFrame()
            
            valid_scores = {k: v for k, v in self.silhouette_scores.items() if v is not None and v != -1}
            if valid_scores:
                self.best_model = max(valid_scores, key=valid_scores.get)
            
            if os.path.exists("outputs/business_insights.json"):
                with open("outputs/business_insights.json", 'r') as f:
                    self.business_insights = json.load(f)
                for insight in self.business_insights:
                    insight['cluster_id'] = int(insight['cluster_id'])
            else:
                self.business_insights = []
            
            self.models_loaded = True
            logger.info(f"Models and data loaded successfully! Best model: {self.best_model} "
                        f"(Silhouette Scores: {self.silhouette_scores})")
            
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.info("Please run the training pipeline first using train_models.py")
            self.models_loaded = False
    
    def calculate_risk_score(self, age, saving, freq, duration):
        """Calculate risk score aligned with new.py"""
        score = (age * 0.5) + (saving / 5000) + (freq * 2.0) - (duration * 0.1)
        return max(10, min(80, round(score, 1)))
    
    def calculate_roi(self, risk_score, freq, inv_type):
        """Calculate expected ROI aligned with new.py"""
        base_roi = 6
        risk_bonus = risk_score / 10
        freq_bonus = freq * 0.2
        inv_type_bonus_map = {
            'stocks': 7, 'mutual': 5, 'crypto': 12, 'gold': 3,
            'fd': 2, 'ppf': 2.5, 'real_estate': 5, 'etf': 5
        }
        inv_type_bonus = inv_type_bonus_map.get(inv_type, 3)
        return max(6, min(30, round(base_roi + risk_bonus + freq_bonus + inv_type_bonus, 2)))
    
    def prepare_input_features(self, age, saving, net_worth, risk_score, tenure, duration, freq, roi, goal, inv_type):
        """Prepare input features for prediction with weighted features"""
        numerical_features = pd.DataFrame({
            'age': [age],
            'saving': [saving],
            'net_worth': [net_worth],
            'risk_score': [risk_score * 1.5],
            'tenure': [tenure],
            'duration': [duration * 2.5],
            'freq': [freq],
            'roi': [roi * 1.5]
        })
        
        categorical_features = pd.DataFrame()
        
        if 'goal' in self.label_encoders:
            try:
                goal_encoded = self.label_encoders['goal'].transform([goal])[0] * 2.5
                categorical_features['goal_encoded'] = [goal_encoded]
            except ValueError:
                logger.warning(f"Unknown goal: {goal}, using default encoding")
                categorical_features['goal_encoded'] = [0]
        
        if 'inv_type' in self.label_encoders:
            try:
                inv_type_encoded = self.label_encoders['inv_type'].transform([inv_type])[0] * 2.5
                categorical_features['inv_type_encoded'] = [inv_type_encoded]
            except ValueError:
                logger.warning(f"Unknown inv_type: {inv_type}, using default encoding")
                categorical_features['inv_type_encoded'] = [0]
        
        input_features = pd.concat([numerical_features, categorical_features], axis=1)
        input_features = input_features[self.metadata['feature_names']]
        scaled_features = self.scaler.transform(input_features)
        pca_features = self.pca.transform(scaled_features)
        return pca_features
    
    def predict_cluster(self, input_features):
        """Predict cluster for input features using the best model"""
        try:
            if self.best_model == 'gmm':
                cluster = self.gmm_model.predict(input_features)[0]
            elif self.best_model == 'dbscan':
                cluster = self.dbscan_model.fit_predict(input_features)[0]
            else:
                cluster = self.kmeans_model.predict(input_features)[0]
            return cluster
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            return None
    
    def get_cluster_description(self, cluster_id, method=None):
        """Get dynamic description for a cluster with duration ranges"""
        cluster_descriptions = {
            0: {
                'name': 'Conservative Low-Risk',
                'description': 'Older investors prioritizing capital preservation with low-risk, stable investments, typically with long durations (60-100 months).',
                'characteristics': [
                    'Low risk tolerance (score 10-30)',
                    'Prefers fixed deposits and PPF',
                    'Retirement-focused',
                    'Long investment duration (60-100 months)',
                    'Top goal: retirement',
                    'Top inv_type: fd, ppf'
                ]
            },
            1: {
                'name': 'Moderate Risk Mid-Wealth',
                'description': 'Middle-aged investors seeking balanced growth with moderate risk, typically with medium durations (30-60 months).',
                'characteristics': [
                    'Moderate risk tolerance (score 25-50)',
                    'Prefers mutual funds and ETFs',
                    'Education or wealth goals',
                    'Medium-term duration (30-60 months)',
                    'Top goal: education, wealth',
                    'Top inv_type: mutual, etf'
                ]
            },
            2: {
                'name': 'High-Risk Tech-Savvy',
                'description': 'Young, tech-savvy investors comfortable with high risk and volatility, typically with short durations (1-24 months).',
                'characteristics': [
                    'High risk tolerance (score 50-80)',
                    'Prefers crypto and stocks',
                    'Wealth or vacation goals',
                    'Short-term duration (1-24 months)',
                    'Top goal: wealth, vacation',
                    'Top inv_type: crypto, stocks'
                ]
            },
            3: {
                'name': 'Balanced Long-term',
                'description': 'Mature investors focused on long-term wealth accumulation with diversified portfolios, typically with long durations (80-120 months).',
                'characteristics': [
                    'Moderate to high net worth',
                    'Prefers real estate and mutual funds',
                    'Retirement or wealth goals',
                    'Long-term duration (80-120 months)',
                    'Top goal: retirement, wealth',
                    'Top inv_type: real_estate, mutual'
                ]
            }
        }
        
        if cluster_id not in cluster_descriptions:
            logger.warning(f"Unexpected cluster ID {cluster_id} encountered. Using default description.")
            return {
                'name': 'Unclassified Investor Group',
                'description': 'Investors with an undefined profile due to unexpected clustering. Please retrain the model or check data consistency.',
                'characteristics': ['Undefined risk tolerance', 'Mixed investment behavior', 'Requires further analysis']
            }
        
        return cluster_descriptions[cluster_id]
    
    def generate_recommendations(self, cluster_id, risk_score, goal, inv_type):
        """Generate tailored investment recommendations"""
        recommendations = []
        
        cluster_insight = next((insight for insight in self.business_insights if insight['cluster_id'] == cluster_id), None)
        if cluster_insight:
            recommendations.append(f"**Primary Strategy:** {cluster_insight['business_strategy']}")

        if risk_score < 30:
            recommendations.extend([
                "Focus on low-risk options like PPF or Fixed Deposits.",
                "Prioritize capital preservation with tax-saving instruments like ELSS."
            ])
        elif 30 <= risk_score < 50:
            recommendations.extend([
                "Diversify with balanced mutual funds or ETFs.",
                "Consider hybrid funds for a mix of equity and debt."
            ])
        else:
            recommendations.extend([
                "Target high-growth assets such as stocks or cryptocurrencies.",
                "Use Systematic Investment Plans (SIPs) for disciplined equity exposure."
            ])

        goal_recommendations = {
            'retirement': ['Invest in long-term equity index funds', 'Balance with debt for stability'],
            'wealth': ['Focus on equity and real estate growth', 'Rebalance annually'],
            'education': ['Use target-date funds aligned with timelines', 'Plan for inflation adjustments'],
            'vacation': ['Opt for short-term debt or liquid funds', 'Avoid high-risk volatility'],
            'emergency': ['Build a high-yield savings account', 'Use liquid mutual funds']
        }
        if goal in goal_recommendations:
            recommendations.extend(goal_recommendations[goal])

        if cluster_id == 0:
            recommendations.append("Prioritize fixed-income securities for long-term stability.")
        elif cluster_id == 1:
            recommendations.append("Maintain a 50/50 equity-debt ratio for balanced growth.")
        elif cluster_id == 2:
            recommendations.append("Leverage high-volatility opportunities with stop-loss limits.")
        elif cluster_id == 3:
            recommendations.append("Focus on diversified asset allocation for long-term wealth.")

        inv_type_strategy = {
            'stocks': "Increase exposure to blue-chip or growth stocks.",
            'crypto': "Limit to 5-10% of portfolio due to volatility.",
            'mutual': "Opt for diversified or sector-specific funds.",
            'gold': "Use as a hedge against inflation.",
            'fd': "Secure short-term savings with competitive rates.",
            'ppf': "Leverage tax benefits for long-term goals.",
            'real_estate': "Consider REITs for diversification.",
            'etf': "Use ETFs for low-cost market exposure."
        }
        recommendations.append(inv_type_strategy.get(inv_type, "Review investment type alignment."))

        return recommendations
    
    def create_interactive_charts(self):
        """Create categorized, larger, and easily understandable interactive charts using Plotly"""
        if self.df_clustered.empty:
            st.warning("No clustered data available for visualization")
            return

        tabs = st.tabs([
            "Portfolio Overview", "Demographic Insights", "Investment Behavior",
            "Investor Group Profiles", "Risk and Return Analysis", "Correlation Studies"
        ])

        with tabs[0]:
            st.markdown("### Portfolio Summary")
            col1, col2 = st.columns(2)
            with col1:
                cluster_counts = self.df_clustered[f'{self.best_model}_cluster'].value_counts().sort_index()
                fig1 = px.pie(
                    names=[self.get_cluster_description(i)['name'] for i in cluster_counts.index],
                    values=cluster_counts.values,
                    title=f"Customer Distribution by Investor Group ({self.best_model.upper()})",
                    hole=0.3,
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                fig1.update_layout(
                    height=450,
                    title_font_size=20,
                    legend_title_text="Investor Groups"
                )
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                roi_by_cluster = self.df_clustered.groupby(f'{self.best_model}_cluster')['roi'].mean().sort_index()
                fig2 = px.bar(
                    x=[self.get_cluster_description(i)['name'] for i in roi_by_cluster.index],
                    y=roi_by_cluster.values,
                    title=f"Average ROI by Investor Group ({self.best_model.upper()})",
                    labels={'x': 'Investor Group', 'y': 'ROI (%)'},
                    color_discrete_sequence=['#003087']
                )
                fig2.update_layout(height=450, title_font_size=20, xaxis_tickangle=45)
                st.plotly_chart(fig2, use_container_width=True)

        with tabs[1]:
            st.markdown("### Customer Demographics")
            col1, col2 = st.columns(2)
            with col1:
                fig3 = px.histogram(
                    self.df_clustered,
                    x='age',
                    title="Age Distribution Across Customers",
                    nbins=20,
                    color_discrete_sequence=['#1f77b4']
                )
                fig3.update_layout(height=450, title_font_size=20, bargap=0.1)
                st.plotly_chart(fig3, use_container_width=True)

            with col2:
                fig4 = px.scatter(
                    self.df_clustered,
                    x='age',
                    y='net_worth',
                    color=f'{self.best_model}_cluster',
                    title=f"Net Worth vs Age by Investor Group ({self.best_model.upper()})",
                    color_continuous_scale='Viridis',
                    labels={'net_worth': 'Net Worth (₹)', 'age': 'Age (Years)'}
                )
                fig4.update_layout(height=450, title_font_size=20)
                st.plotly_chart(fig4, use_container_width=True)

        with tabs[2]:
            st.markdown("### Investment Patterns")
            col1, col2 = st.columns(2)
            with col1:
                fig6 = px.bar(
                    self.df_clustered['inv_type'].value_counts().sort_values(ascending=False),
                    title="Popularity of Investment Types",
                    color_discrete_sequence=['#003087']
                )
                fig6.update_layout(height=450, title_font_size=20, xaxis_tickangle=45)
                st.plotly_chart(fig6, use_container_width=True)

            with col2:
                fig7 = px.scatter(
                    self.df_clustered,
                    x='freq',
                    y='roi',
                    color=f'{self.best_model}_cluster',
                    title=f"Investment Frequency vs ROI ({self.best_model.upper()})",
                    color_continuous_scale='Viridis',
                    labels={'freq': 'Frequency (per year)', 'roi': 'ROI (%)'}
                )
                fig7.update_layout(height=450, title_font_size=20)
                st.plotly_chart(fig7, use_container_width=True)

        with tabs[3]:
            st.markdown("### Investor Group Insights")
            col1, col2 = st.columns(2)
            with col1:
                goal_pivot = self.df_clustered.groupby([f'{self.best_model}_cluster', 'goal']).size().unstack(fill_value=0)
                fig9 = go.Figure()
                for goal in goal_pivot.columns:
                    fig9.add_trace(go.Bar(
                        x=[self.get_cluster_description(i)['name'] for i in goal_pivot.index],
                        y=goal_pivot[goal],
                        name=goal.capitalize()
                    ))
                fig9.update_layout(barmode='stack', title=f"Goal Distribution by Investor Group ({self.best_model.upper()})", height=450, title_font_size=20)
                st.plotly_chart(fig9, use_container_width=True)

            with col2:
                fig10 = px.box(
                    self.df_clustered,
                    x='goal',
                    y='net_worth',
                    title="Net Worth Distribution by Financial Goal",
                    color_discrete_sequence=['#003087']
                )
                fig10.update_layout(height=450, title_font_size=20)
                st.plotly_chart(fig10, use_container_width=True)

        with tabs[4]:
            st.markdown("### Risk and Return Metrics")
            col1, col2 = st.columns(2)
            with col1:
                fig11 = px.violin(
                    self.df_clustered,
                    y='risk_score',
                    color=f'{self.best_model}_cluster',
                    title=f"Risk Score Distribution by Investor Group ({self.best_model.upper()})",
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                fig11.update_layout(height=450, title_font_size=20)
                st.plotly_chart(fig11, use_container_width=True)

            with col2:
                fig12 = px.line(
                    self.df_clustered.groupby('age')['risk_score'].mean().reset_index(),
                    x='age',
                    y='risk_score',
                    title="Average Risk Score by Age",
                    labels={'age': 'Age (Years)', 'risk_score': 'Risk Score'}
                )
                fig12.update_layout(height=450, title_font_size=20)
                st.plotly_chart(fig12, use_container_width=True)

        with tabs[5]:
            st.markdown("### Data Correlations")
            col1, col2 = st.columns(2)
            with col1:
                numerical_cols = ['age', 'saving', 'net_worth', 'risk_score', 'tenure', 'duration', 'freq', 'roi']
                corr_matrix = self.df_clustered[numerical_cols].corr()
                fig14 = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1
                ))
                fig14.update_layout(title="Correlation Between Numerical Features", height=450)
                st.plotly_chart(fig14, use_container_width=True)

            with col2:
                fig15 = px.scatter(
                    self.df_clustered,
                    x='saving',
                    y='net_worth',
                    color=f'{self.best_model}_cluster',
                    title=f"Saving vs Net Worth by Investor Group ({self.best_model.upper()})",
                    color_continuous_scale='Viridis',
                    labels={'saving': 'Monthly Saving (₹)', 'net_worth': 'Net Worth (₹)'}
                )
                fig15.update_layout(height=450, title_font_size=20)
                st.plotly_chart(fig15, use_container_width=True)
    
    def display_cluster_analysis(self):
        """Display comprehensive cluster analysis"""
        if self.df_clustered.empty:
            st.warning("No clustered data available for analysis")
            return
        
        st.subheader(f"📊 Investor Group Analysis Overview ({self.best_model.upper()})")
        
        for cluster_id in sorted(self.df_clustered[f'{self.best_model}_cluster'].unique()):
            cluster_data = self.df_clustered[self.df_clustered[f'{self.best_model}_cluster'] == cluster_id]
            cluster_desc = self.get_cluster_description(cluster_id)
            
            with st.expander(f"{cluster_desc['name']} (n={len(cluster_data)})"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("**Key Metrics**")
                    metrics = [
                        ("Average Age", f"{cluster_data['age'].mean():.1f} years"),
                        ("Average Risk Score", f"{cluster_data['risk_score'].mean():.1f}"),
                        ("Average ROI", f"{cluster_data['roi'].mean():.1f}%"),
                        ("Average Duration", f"{cluster_data['duration'].mean():.1f} months")
                    ]
                    for label, value in metrics:
                        st.markdown(f'<div class="metric-card"><strong>{label}:</strong> {value}</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**Profile**")
                    st.markdown(f'<div class="cluster-description">{cluster_desc["description"]}</div>', unsafe_allow_html=True)
                    st.markdown("**Characteristics**")
                    for char in cluster_desc['characteristics']:
                        st.markdown(f"✓ {char}")
    
    def customer_segmentation_page(self):
        """Main customer segmentation page"""
        st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=200)
        
        st.markdown('<h1 class="main-header">Investment Customer Segmentation 💹</h1>', unsafe_allow_html=True)
        
        if not self.models_loaded:
            st.error("⚠️ Models not loaded. Please run train_models.py first!")
            return
        
        st.markdown("""
        <div class="welcome-box">
            <h3>Welcome to IBM's Investment Analytics</h3>
            <p>Discover your investment profile and get personalized recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Enter Your Details")
        with st.form("customer_form"):
            col1, col2 = st.columns(2)
            
            col1.header("Personal Details")
            age = col1.number_input("Age", min_value=18, max_value=75, value=30)
            net_worth = col1.number_input("Net Worth (₹)", min_value=50000, max_value=10000000, value=500000, step=50000)
            
            col2.header("Investment Profile")
            saving = col2.number_input("Monthly Saving (₹)", min_value=2000, max_value=100000, value=20000, step=1000)
            tenure = col2.slider("Years of Investing Experience", 0, 30, 5)
            duration = col2.slider("Average Investment Duration (months)", 1, 120, 24)
            freq = col2.slider("Investment Frequency (per year)", 1, 30, 5)
            
            col1.header("Investment Preferences")
            goal = col1.selectbox("Primary Financial Goal",
                                 ['retirement', 'wealth', 'education', 'vacation', 'emergency'])
            inv_type = col2.selectbox("Preferred Investment Type",
                                     ['stocks', 'mutual', 'crypto', 'gold', 'fd', 'ppf', 'real_estate', 'etf'])
            
            submit_button = st.form_submit_button("🔍 Analyze My Profile", type="primary")
        
        if submit_button:
            risk_score = self.calculate_risk_score(age, saving, freq, duration)
            roi = self.calculate_roi(risk_score, freq, inv_type)
            
            input_features = self.prepare_input_features(
                age, saving, net_worth, risk_score, tenure, duration, freq, roi, goal, inv_type
            )
            
            cluster = self.predict_cluster(input_features)
            
            if cluster is not None:
                cluster_desc = self.get_cluster_description(cluster)
                
                st.markdown("## 📈 Your Investment Profile")
                col1, col2, col3 = st.columns(3)
                col1.markdown(f'<div class="metric-card"><strong>Risk Score</strong><br>{risk_score}/100</div>', unsafe_allow_html=True)
                col2.markdown(f'<div class="metric-card"><strong>Expected ROI</strong><br>{roi}%</div>', unsafe_allow_html=True)
                col3.markdown(f'<div class="metric-card"><strong>Investment Horizon</strong><br>{duration} months</div>', unsafe_allow_html=True)
                
                st.markdown("## 🎯 Your Investor Profile")
                st.markdown(f'<div class="cluster-description">', unsafe_allow_html=True)
                st.markdown(f"### {cluster_desc['name']}")
                st.markdown(f"**Group ID:** {cluster}")
                st.markdown(f"**Description:** {cluster_desc['description']}")
                st.markdown(f"**Model Used:** {self.best_model.upper()} (Silhouette Score: {self.silhouette_scores[self.best_model]:.3f})")
                st.markdown('</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                for i, char in enumerate(cluster_desc['characteristics']):
                    with col1 if i % 2 == 0 else col2:
                        st.markdown(f"✓ {char}")
                
                st.markdown("## 💡 Personalized Recommendations")
                recommendations = self.generate_recommendations(cluster, risk_score, goal, inv_type)
                st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                for rec in recommendations:
                    st.markdown(f"• {rec}")
                st.markdown('</div>', unsafe_allow_html=True)
    
    def analytics_dashboard_page(self):
        """Analytics dashboard with visualizations"""
        st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=200)
        
        st.markdown('<h1 class="main-header">📊 Analytics Dashboard</h1>', unsafe_allow_html=True)
        
        if not self.models_loaded or self.df_clustered.empty:
            st.warning("⚠️ No data available for analytics. Please run the training pipeline first.")
            return
        
        st.markdown("## 📈 Portfolio Overview")
        col1, col2, col3, col4 = st.columns(4)
        metrics = [
            ("Total Customers", len(self.df_clustered)),
            ("Average Risk Score", f"{self.df_clustered['risk_score'].mean():.1f}"),
            ("Average ROI", f"{self.df_clustered['roi'].mean():.1f}%"),
            ("Investor Groups Identified", len(self.df_clustered[f'{self.best_model}_cluster'].unique()))
        ]
        for i, (label, value) in enumerate(metrics):
            with [col1, col2, col3, col4][i]:
                st.markdown(f'<div class="stat-card"><strong>{label}</strong><br>{value}</div>', unsafe_allow_html=True)
        
        st.markdown("## 📊 Investment Insights")
        self.create_interactive_charts()
        
        self.display_cluster_analysis()
        
        if self.business_insights:
            st.markdown("## 💼 Strategic Insights")
            cluster_segment_names = {
                0: "Conservative Low-Risk",
                1: "Moderate Risk Mid-Wealth",
                2: "High-Risk Tech-Savvy",
                3: "Balanced Long-term"
            }
            
            for insight in self.business_insights:
                cluster_id = insight['cluster_id']
                segment_name = cluster_segment_names.get(cluster_id, "Unclassified Investor Group")
                
                with st.expander(segment_name):
                    st.markdown(f'<div class="metric-card"><strong>Size:</strong> {insight["size"]} customers</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-card"><strong>Strategy:</strong> {insight["business_strategy"]}</div>', unsafe_allow_html=True)
                    st.markdown("**Key Characteristics**")
                    for key, value in insight['key_characteristics'].items():
                        st.markdown(f"• {key.replace('_', ' ').title()}: {value}")
    
    def model_insights_page(self):
        """Model insights page with comparison of all clustering methods"""
        st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=200)
        
        st.markdown('<h1 class="main-header">🤖 Model Insights</h1>', unsafe_allow_html=True)
        
        if not self.models_loaded:
            st.warning("⚠️ Models not loaded. Please run the training pipeline first.")
            return
        
        st.markdown("## 🔧 Model Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### KMeans Model")
            if 'best_k' in self.metadata:
                st.markdown(f'<div class="metric-card"><strong>Optimal Groups:</strong> {self.metadata["best_k"]}</div>', unsafe_allow_html=True)
            if self.silhouette_scores['kmeans'] is not None:
                st.markdown(f'<div class="metric-card"><strong>Silhouette Score:</strong> {self.silhouette_scores["kmeans"]:.3f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><strong>Algorithm:</strong> K-Means Clustering</div>', unsafe_allow_html=True)
            st.markdown("""
                **Description**: Partitions customers into clusters by minimizing distance to centroids, 
                assuming spherical clusters. Optimized with n_init=30 and PCA (5 components).
            """)
        
        with col2:
            st.markdown("### GMM Model")
            if 'best_k' in self.metadata:
                st.markdown(f'<div class="metric-card"><strong>Optimal Groups:</strong> {self.metadata["best_k"]}</div>', unsafe_allow_html=True)
            if self.silhouette_scores['gmm'] is not None:
                st.markdown(f'<div class="metric-card"><strong>Silhouette Score:</strong> {self.silhouette_scores["gmm"]:.3f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><strong>Algorithm:</strong> Gaussian Mixture Model</div>', unsafe_allow_html=True)
            st.markdown("""
                **Description**: Uses probabilistic assignments with tied covariance, modeling complex 
                cluster shapes. Optimized with max_iter=200 and PCA.
            """)
            
            st.markdown("### DBSCAN Model")
            if self.silhouette_scores['dbscan'] is not None:
                st.markdown(f'<div class="metric-card"><strong>Silhouette Score:</strong> {self.silhouette_scores["dbscan"]:.3f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><strong>Algorithm:</strong> DBSCAN</div>', unsafe_allow_html=True)
            st.markdown("""
                **Description**: Density-based clustering identifying dense regions, handling noise and 
                irregular shapes. Uses eps=0.5 and min_samples=5 with PCA.
            """)
        
        st.markdown("### Model Comparison")
        if any(score is not None and score != -1 for score in self.silhouette_scores.values()):
            valid_scores = {k: v for k, v in self.silhouette_scores.items() if v is not None and v != -1}
            best_model = max(valid_scores, key=valid_scores.get)
            st.markdown(f"""
                Silhouette scores reflect clustering quality (range: -1 to 1):
                - **KMeans ({self.silhouette_scores['kmeans']:.3f})**: Efficient for spherical clusters.
                - **GMM ({self.silhouette_scores['gmm']:.3f})**: Captures complex, elliptical clusters.
                - **DBSCAN ({self.silhouette_scores['dbscan']:.3f})**: Density-based, handles noise but may vary in cluster count.
                
                The **{best_model.upper()}** model is selected due to its highest silhouette score 
                ({self.silhouette_scores[best_model]:.3f}), ensuring optimal cluster cohesion and separation.
                All models use PCA (5 components) and weighted features for enhanced performance.
            """)
            fig = px.bar(
                x=list(valid_scores.keys()),
                y=list(valid_scores.values()),
                title='Silhouette Score Comparison Across Models',
                labels={'x': 'Model', 'y': 'Silhouette Score'},
                color=list(valid_scores.keys()),
                color_discrete_map={
                    'kmeans': '#003087',
                    'gmm': '#1f77b4',
                    'dbscan': '#2ca02c'
                }
            )
            fig.update_layout(height=300, title_font_size=20)
            st.plotly_chart(fig, use_container_width=True)
            
            fig2 = go.Figure()
            for method, color in zip(['kmeans', 'gmm', 'dbscan'], 
                                   ['#003087', '#1f77b4', '#2ca02c']):
                if self.silhouette_scores[method] == -1:
                    continue
                counts = self.df_clustered[f'{method}_cluster'].value_counts().sort_index()
                fig2.add_trace(go.Bar(
                    x=counts.index,
                    y=counts.values,
                    name=method.capitalize(),
                    marker_color=color
                ))
            fig2.update_layout(
                barmode='group',
                title='Cluster Size Comparison Across Models',
                xaxis_title='Cluster ID',
                yaxis_title='Number of Customers',
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No valid silhouette scores available. Please run train_models.py to compute them.")
        
        st.markdown("## 🔍 Optimal Investor Group Analysis")
        if os.path.exists("outputs/optimal_clusters_analysis.png"):
            st.image("outputs/optimal_clusters_analysis.png", caption="Optimal Investor Groups Analysis")
            st.markdown("""
                The elbow curve shows inertia for varying cluster counts. The optimal number of groups (k=4) balances 
                compactness and complexity, enhanced by PCA and feature weighting.
            """)
        else:
            st.warning("Optimal clusters analysis image not found. Please run train_models.py to generate it.")
        
        st.markdown("## 📏 Silhouette Score Analysis")
        if any(score is not None and score != -1 for score in self.silhouette_scores.values()):
            st.markdown(f"""
                Silhouette scores measure cluster cohesion and separation:
                - **KMeans**: {self.silhouette_scores['kmeans']:.3f}
                - **GMM**: {self.silhouette_scores['gmm']:.3f}
                - **DBSCAN**: {self.silhouette_scores['dbscan']:.3f}
                
                Scores above 0.5 indicate robust clustering. The {best_model.upper()} model is used for predictions, 
                leveraging PCA and weighted features (duration: 2.5, risk_score/roi: 1.5, goal/inv_type: 2.5).
            """)
        else:
            st.warning("Silhouette scores not available. Please run train_models.py to compute them.")
        
        st.markdown("## 📊 Feature Importance Analysis")
        if 'feature_names' in self.metadata:
            features_df = pd.DataFrame({
                'Feature': self.metadata['feature_names'],
                'Type': ['Numerical' if 'encoded' not in f else 'Categorical' for f in self.metadata['feature_names']],
                'Weight': [2.5 if f in ['duration', 'goal_encoded', 'inv_type_encoded'] else 1.5 if f in ['risk_score', 'roi'] else 1.0 for f in self.metadata['feature_names']]
            })
            fig = px.bar(
                features_df,
                x='Feature',
                y='Weight',
                color='Type',
                title='Features and Weights Used in Clustering',
                color_discrete_map={'Numerical': '#003087', 'Categorical': '#1f77b4'}
            )
            fig.update_layout(showlegend=True, height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
                The clustering model uses 10 weighted features:
                - **Age (1.0)**: Influences risk tolerance.
                - **Saving (1.0)**: Reflects financial commitment.
                - **Net Worth (1.0)**: Guides allocation strategies.
                - **Risk Score (1.5)**: Quantifies risk tolerance.
                - **Tenure (1.0)**: Measures experience.
                - **Duration (2.5)**: Key for goal alignment.
                - **Freq (1.0)**: Indicates engagement.
                - **ROI (1.5)**: Expected return.
                - **Goal Encoded (2.5)**: Drives planning.
                - **Inv Type Encoded (2.5)**: Shapes diversification.

                Weights and PCA (5 components) enhance cluster separation.
            """)
    
    def run(self):
        """Run the Streamlit application"""
        st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=100)
        st.sidebar.markdown("### Investment Analytics")
        page = st.sidebar.radio(
            "Navigate",
            ["Customer Segmentation", "Analytics Dashboard", "Model Insights"]
        )
        
        if page == "Customer Segmentation":
            self.customer_segmentation_page()
        elif page == "Analytics Dashboard":
            self.analytics_dashboard_page()
        else:
            self.model_insights_page()

def main():
    """Main function to initialize and run the app"""
    app = InvestmentSegmentationApp()
    app.run()

if __name__ == "__main__":
    main()