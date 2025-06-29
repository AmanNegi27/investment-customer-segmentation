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
        self.load_models_and_data()
        
    def load_models_and_data(self):
        """Load trained models and preprocessors"""
        try:
            self.scaler = joblib.load("models/scaler.pkl")
            self.kmeans_model = joblib.load("models/kmeans_model.pkl")
            self.gmm_model = joblib.load("models/gmm_model.pkl")
            self.label_encoders = joblib.load("models/label_encoders.pkl")
            
            if os.path.exists("models/model_metadata.json"):
                with open("models/model_metadata.json", 'r') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {}
            
            if os.path.exists("outputs/cluster_profiles.json"):
                with open("outputs/cluster_profiles.json", 'r') as f:
                    self.cluster_profiles = json.load(f)
            else:
                self.cluster_profiles = {}
            
            if os.path.exists("outputs/business_insights.json"):
                with open("outputs/business_insights.json", 'r') as f:
                    self.business_insights = json.load(f)
            else:
                self.business_insights = []
            
            if os.path.exists("outputs/clustered_dataset.csv"):
                self.df_clustered = pd.read_csv("outputs/clustered_dataset.csv")
            else:
                self.df_clustered = pd.DataFrame()
            
            self.models_loaded = True
            logger.info("Models and data loaded successfully!")
            
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.info("Please run the training pipeline first using train_models.py")
            self.models_loaded = False
    
    def calculate_risk_score(self, age, saving, freq, duration):
        """Calculate risk score based on customer inputs"""
        score = (age * 0.3) + (saving / 8000) + (freq * 1.2) - (duration * 0.1)
        return max(0, min(100, round(score, 1)))
    
    def calculate_roi(self, risk_score, freq, inv_type):
        """Calculate expected ROI based on risk score and investment type"""
        base_roi = 5
        risk_bonus = risk_score / 20
        freq_bonus = freq * 0.1
        
        inv_type_bonus_map = {
            'stocks': 5, 'mutual': 4, 'crypto': 8, 'gold': 2,
            'fd': 1.5, 'ppf': 2, 'bonds': 2.5, 'real_estate': 3,
            'etf': 4, 'nps': 2
        }
        inv_type_bonus = inv_type_bonus_map.get(inv_type, 2)
        
        return round(base_roi + risk_bonus + freq_bonus + inv_type_bonus, 2)
    
    def prepare_input_features(self, age, saving, net_worth, risk_score, tenure, duration, freq, roi, goal, inv_type):
        """Prepare input features for prediction"""
        numerical_features = pd.DataFrame({
            'age': [age],
            'saving': [saving],
            'net_worth': [net_worth],
            'risk_score': [risk_score],
            'tenure': [tenure],
            'duration': [duration],
            'freq': [freq],
            'roi': [roi]
        })
        
        categorical_features = pd.DataFrame()
        
        if 'goal' in self.label_encoders:
            try:
                goal_encoded = self.label_encoders['goal'].transform([goal])[0]
                categorical_features['goal_encoded'] = [goal_encoded]
            except ValueError:
                categorical_features['goal_encoded'] = [0]
        
        if 'inv_type' in self.label_encoders:
            try:
                inv_type_encoded = self.label_encoders['inv_type'].transform([inv_type])[0]
                categorical_features['inv_type_encoded'] = [inv_type_encoded]
            except ValueError:
                categorical_features['inv_type_encoded'] = [0]
        
        if categorical_features.empty:
            input_features = numerical_features
        else:
            input_features = pd.concat([numerical_features, categorical_features], axis=1)
        
        return input_features
    
    def predict_cluster(self, input_features):
        """Predict cluster for input features"""
        try:
            scaled_features = self.scaler.transform(input_features)
            kmeans_cluster = self.kmeans_model.predict(scaled_features)[0]
            gmm_cluster = self.gmm_model.predict(scaled_features)[0]
            return kmeans_cluster, gmm_cluster
            
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            return None, None
    
    def get_cluster_description(self, cluster_id, method='kmeans'):
        """Get description for a cluster"""
        cluster_descriptions = {
            0: {
                'name': '🛡️ Conservative Investors',
                'description': 'Low-risk, stability-focused investors who prioritize capital preservation over high returns.',
                'characteristics': ['Lower risk tolerance', 'Stable income', 'Long-term focused', 'Prefer guaranteed returns']
            },
            1: {
                'name': '📊 Balanced Investors',
                'description': 'Moderate risk-taking investors seeking balanced growth with reasonable security.',
                'characteristics': ['Moderate risk tolerance', 'Diversified portfolio', 'Mixed investment horizon', 'Growth with safety']
            },
            2: {
                'name': '🚀 Aggressive Investors',
                'description': 'High-risk, high-reward seeking investors comfortable with market volatility.',
                'characteristics': ['High risk tolerance', 'Growth-oriented', 'Tech-savvy', 'Active trading']
            },
            3: {
                'name': '💼 Wealth Builders',
                'description': 'Long-term wealth accumulation focused investors with substantial assets.',
                'characteristics': ['High net worth', 'Long-term planning', 'Wealth preservation', 'Strategic investing']
            }
        }
        
        return cluster_descriptions.get(cluster_id, {
            'name': f'Cluster {cluster_id}',
            'description': 'Investment profile not fully characterized',
            'characteristics': ['Mixed characteristics']
        })
    
    def generate_recommendations(self, cluster_id, risk_score, goal, inv_type):
        """Generate investment recommendations based on cluster and profile"""
        recommendations = []
        
        cluster_insight = None
        for insight in self.business_insights:
            if insight['cluster_id'] == cluster_id:
                cluster_insight = insight
                break
        
        if cluster_insight:
            recommendations.append(f"**Primary Strategy:** {cluster_insight['business_strategy']}")
        
        if risk_score < 25:
            recommendations.extend([
                "Consider low-risk instruments: PPF, FDs, Government Bonds",
                "Focus on capital preservation and tax-saving investments",
                "Maintain emergency fund in liquid assets"
            ])
        elif risk_score < 50:
            recommendations.extend([
                "Diversify with mutual funds and ETFs",
                "Consider balanced funds and hybrid instruments",
                "Mix of equity and debt for optimal returns"
            ])
        else:
            recommendations.extend([
                "Explore high-growth opportunities: Stocks, Crypto, REITs",
                "Consider systematic investment plans (SIPs)",
                "Monitor market trends for active portfolio management"
            ])
        
        goal_recommendations = {
            'retirement': ['Maximize EPF contributions', 'Consider NPS for tax benefits', 'Build long-term equity portfolio'],
            'wealth': ['Focus on equity investments', 'Consider real estate', 'Explore international diversification'],
            'education': ['Target date funds', 'Conservative debt funds', 'Child education plans'],
            'vacation': ['Short-term debt funds', 'Liquid funds', 'Conservative hybrid funds'],
            'emergency': ['High-yield savings accounts', 'Liquid mutual funds', 'Ultra-short duration funds']
        }
        
        if goal in goal_recommendations:
            recommendations.extend(goal_recommendations[goal])
        
        return recommendations
    
    def create_interactive_charts(self):
        """Create enhanced interactive charts using Plotly"""
        if self.df_clustered.empty:
            st.warning("No clustered data available for visualization")
            return
        
        # Create subplot layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Risk Score vs ROI by Cluster",
                "Age vs Net Worth Distribution",
                "Investment Type Distribution",
                "Cluster Distribution"
            ),
            specs=[[{"type": "scatter"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )
        
        # 1. Risk vs ROI scatter plot
        scatter = go.Scatter(
            x=self.df_clustered['risk_score'],
            y=self.df_clustered['roi'],
            mode='markers',
            marker=dict(
                color=self.df_clustered['kmeans_cluster'],
                colorscale='Viridis',
                showscale=True,
                size=10,
                opacity=0.7
            ),
            text=self.df_clustered.apply(lambda x: f"Age: {x['age']}<br>Goal: {x['goal']}", axis=1),
            hoverinfo='text'
        )
        fig.add_trace(scatter, row=1, col=1)
        
        # 2. Age vs Net Worth histogram
        hist = go.Histogram2d(
            x=self.df_clustered['age'],
            y=self.df_clustered['net_worth'] / 100000,
            colorscale='Blues',
            nbinsx=20,
            nbinsy=20,
            hoverinfo='x+y+z'
        )
        fig.add_trace(hist, row=1, col=2)
        
        # 3. Investment Type Distribution
        inv_type_counts = self.df_clustered['inv_type'].value_counts()
        bar = go.Bar(
            x=inv_type_counts.index,
            y=inv_type_counts.values,
            marker=dict(color=inv_type_counts.values, colorscale='Portland')
        )
        fig.add_trace(bar, row=2, col=1)
        
        # 4. Cluster distribution
        cluster_counts = self.df_clustered['kmeans_cluster'].value_counts().sort_index()
        pie = go.Pie(
            labels=[f'Cluster {i}' for i in cluster_counts.index],
            values=cluster_counts.values,
            hole=0.3,
            marker=dict(colors=px.colors.qualitative.Plotly)
        )
        fig.add_trace(pie, row=2, col=2)
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Customer Segmentation Analytics",
            title_x=0.5,
            title_font=dict(size=20, color='#003087')
        )
        fig.update_xaxes(title_text="Risk Score", row=1, col=1)
        fig.update_yaxes(title_text="ROI (%)", row=1, col=1)
        fig.update_xaxes(title_text="Age", row=1, col=2)
        fig.update_yaxes(title_text="Net Worth (Lakhs)", row=1, col=2)
        fig.update_xaxes(title_text="Investment Type", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_cluster_analysis(self):
        """Display comprehensive cluster analysis with improved visuals"""
        if self.df_clustered.empty:
            st.warning("No clustered data available for analysis")
            return
        
        st.subheader("📊 Cluster Analysis Overview")
        
        for cluster_id in sorted(self.df_clustered['kmeans_cluster'].unique()):
            cluster_data = self.df_clustered[self.df_clustered['kmeans_cluster'] == cluster_id]
            cluster_desc = self.get_cluster_description(cluster_id)
            
            with st.expander(f"{cluster_desc['name']} (n={len(cluster_data)})"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("**Key Metrics**")
                    metrics = [
                        ("Average Age", f"{cluster_data['age'].mean():.1f} years"),
                        ("Average Risk Score", f"{cluster_data['risk_score'].mean():.1f}"),
                        ("Average ROI", f"{cluster_data['roi'].mean():.1f}%"),
                        ("Average Net Worth", f"₹{cluster_data['net_worth'].mean()/100000:.1f}L")
                    ]
                    for label, value in metrics:
                        st.markdown(f'<div class="metric-card"><strong>{label}:</strong> {value}</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**Profile**")
                    st.markdown(f'<div class="cluster-description">{cluster_desc["description"]}</div>', unsafe_allow_html=True)
                    st.markdown("**Characteristics**")
                    for char in cluster_desc['characteristics']:
                        st.markdown(f"✓ {char}")
                
                # Visual: Top goals and investment types
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown("**Top Goals**")
                    top_goals = cluster_data['goal'].value_counts().head(3)
                    fig_goals = go.Figure(data=[
                        go.Bar(x=top_goals.values, y=top_goals.index, orientation='h',
                               marker=dict(color='#003087'))
                    ])
                    fig_goals.update_layout(height=200, margin=dict(t=0, b=0))
                    st.plotly_chart(fig_goals, use_container_width=True)
                
                with col4:
                    st.markdown("**Top Investment Types**")
                    top_inv = cluster_data['inv_type'].value_counts().head(3)
                    fig_inv = go.Figure(data=[
                        go.Bar(x=top_inv.values, y=top_inv.index, orientation='h',
                               marker=dict(color='#1f77b4'))
                    ])
                    fig_inv.update_layout(height=200, margin=dict(t=0, b=0))
                    st.plotly_chart(fig_inv, use_container_width=True)
    
    def customer_segmentation_page(self):
        """Main customer segmentation page with enhanced UI"""
        # IBM Logo
        st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=200)
        
        st.markdown('<h1 class="main-header">🎯 Investment Customer Segmentation</h1>', unsafe_allow_html=True)
        
        if not self.models_loaded:
            st.error("⚠️ Models not loaded. Please run train_models.py first!")
            return
        
        # Welcome section
        st.markdown("""
        <div class="welcome-box">
            <h3>Welcome to IBM's Investment Analytics</h3>
            <p>Discover your investment profile and get personalized recommendations powered by advanced machine learning.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Input form
        st.markdown("### Enter Your Details")
        with st.form("customer_form"):
            col1, col2 = st.columns(2)
            
            col1.header("Personal Details")
            age = col1.number_input("Age", min_value=18, max_value=75, value=30)
            net_worth = col1.number_input("Net Worth (₹)", min_value=10000, max_value=10000000, value=500000, step=50000)
            
            col2.header("Investment Profile")
            saving = col2.number_input("Monthly Saving (₹)", min_value=1000, max_value=100000, value=20000, step=1000)
            tenure = col2.slider("Years of Investing Experience", 0, 30, 5)
            duration = col2.slider("Average Investment Duration (months)", 1, 120, 24)
            freq = col2.slider("Investment Frequency (per year)", 1, 30, 5)
            
            col1.header("Investment Preferences")
            goal = col1.selectbox("Primary Financial Goal", 
                                 ['retirement', 'wealth', 'education', 'vacation', 'emergency'])
            inv_type = col2.selectbox("Preferred Investment Type", 
                                     ['stocks', 'mutual', 'crypto', 'gold', 'fd', 'ppf', 'bonds', 'real_estate', 'etf', 'nps'])
            
            submit_button = st.form_submit_button("🔍 Analyze My Profile", type="primary")
        
        if submit_button:
            # Calculate metrics
            risk_score = self.calculate_risk_score(age, saving, freq, duration)
            roi = self.calculate_roi(risk_score, freq, inv_type)
            
            # Display metrics
            st.markdown("## 📈 Your Investment Profile")
            col1, col2, col3 = st.columns(3)
            col1.markdown(f'<div class="metric-card"><strong>Risk Score</strong><br>{risk_score}/100</div>', unsafe_allow_html=True)
            col2.markdown(f'<div class="metric-card"><strong>Expected ROI</strong><br>{roi}%</div>', unsafe_allow_html=True)
            col3.markdown(f'<div class="metric-card"><strong>Investment Horizon</strong><br>{duration} months</div>', unsafe_allow_html=True)
            
            # Predict clusters
            input_features = self.prepare_input_features(
                age, saving, net_worth, risk_score, tenure, duration, freq, roi, goal, inv_type
            )
            kmeans_cluster, gmm_cluster = self.predict_cluster(input_features)
            
            if kmeans_cluster is not None:
                cluster_desc = self.get_cluster_description(kmeans_cluster)
                
                st.markdown("## 🎯 Your Investor Profile")
                st.markdown(f'<div class="cluster-description">', unsafe_allow_html=True)
                st.markdown(f"### {cluster_desc['name']}")
                st.markdown(f"**Cluster ID:** {kmeans_cluster}")
                st.markdown(f"**Description:** {cluster_desc['description']}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                for i, char in enumerate(cluster_desc['characteristics']):
                    with col1 if i % 2 == 0 else col2:
                        st.markdown(f"✓ {char}")
                
                st.markdown("## 💡 Personalized Recommendations")
                recommendations = self.generate_recommendations(kmeans_cluster, risk_score, goal, inv_type)
                st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                for rec in recommendations:
                    st.markdown(f"• {rec}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                with st.expander("🔍 Model Details"):
                    col1, col2 = st.columns(2)
                    col1.metric("KMeans Cluster", kmeans_cluster)
                    col2.metric("GMM Cluster", gmm_cluster)
                    if 'kmeans_silhouette' in self.metadata:
                        st.info(f"Model Quality - KMeans Silhouette Score: {self.metadata['kmeans_silhouette']:.3f}")
    
    def analytics_dashboard_page(self):
        """Analytics dashboard with enhanced visualizations"""
        st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=200)
        
        st.markdown('<h1 class="main-header">📊 Analytics Dashboard</h1>', unsafe_allow_html=True)
        
        if not self.models_loaded or self.df_clustered.empty:
            st.warning("⚠️ No data available for analytics. Please run the training pipeline first.")
            return
        
        # Key metrics
        st.markdown("## 📈 Portfolio Overview")
        col1, col2, col3, col4 = st.columns(4)
        metrics = [
            ("Total Customers", len(self.df_clustered)),
            ("Average Risk Score", f"{self.df_clustered['risk_score'].mean():.1f}"),
            ("Average ROI", f"{self.df_clustered['roi'].mean():.1f}%"),
            ("Clusters Identified", len(self.df_clustered['kmeans_cluster'].unique()))
        ]
        for i, (label, value) in enumerate(metrics):
            with [col1, col2, col3, col4][i]:
                st.markdown(f'<div class="stat-card"><strong>{label}</strong><br>{value}</div>', unsafe_allow_html=True)
        
        # Visualizations
        st.markdown("## 📊 Investment Insights")
        self.create_interactive_charts()
        
        # Cluster analysis
        self.display_cluster_analysis()
        
        # Business insights
        if self.business_insights:
            st.markdown("## 💼 Strategic Insights")
            for insight in self.business_insights:
                with st.expander(f"Cluster {insight['cluster_id']}: {insight['segment_type']}"):
                    st.markdown(f'<div class="metric-card"><strong>Size:</strong> {insight["size"]} customers</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-card"><strong>Strategy:</strong> {insight["business_strategy"]}</div>', unsafe_allow_html=True)
                    st.markdown("**Key Characteristics**")
                    for key, value in insight['key_characteristics'].items():
                        st.markdown(f"• {key.replace('_', ' ').title()}: {value}")
    
    def model_insights_page(self):
        """Model insights page with IBM branding"""
        st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=200)
        
        st.markdown('<h1 class="main-header">🤖 Model Insights</h1>', unsafe_allow_html=True)
        
        if not self.models_loaded:
            st.warning("⚠️ Models not loaded. Please run the training pipeline first.")
            return
        
        # Model metadata
        st.markdown("## 🔧 Model Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### KMeans Model")
            if 'best_k' in self.metadata:
                st.markdown(f'<div class="metric-card"><strong>Optimal Clusters:</strong> {self.metadata["best_k"]}</div>', unsafe_allow_html=True)
            if 'kmeans_silhouette' in self.metadata:
                st.markdown(f'<div class="metric-card"><strong>Silhouette Score:</strong> {self.metadata["kmeans_silhouette"]:.3f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-card"><strong>Algorithm:</strong> K-Means Clustering</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### GMM Model")
            if 'best_k' in self.metadata:
                st.markdown(f'<div class="metric-card"><strong>Components:</strong> {self.metadata["best_k"]}</div>', unsafe_allow_html=True)
            if 'gmm_silhouette' in self.metadata:
                st.markdown(f'<div class="metric-card"><strong>Silhouette Score:</strong> {self.metadata["gmm_silhouette"]:.3f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-card"><strong>Algorithm:</strong> Gaussian Mixture Model</div>', unsafe_allow_html=True)
        
        # Feature importance visualization
        if 'feature_names' in self.metadata:
            st.markdown("## 📊 Feature Analysis")
            features_df = pd.DataFrame({
                'Feature': self.metadata['feature_names'],
                'Type': ['Numerical' if 'encoded' not in f else 'Categorical' for f in self.metadata['feature_names']]
            })
            fig = px.bar(
                features_df,
                x='Feature',
                y=[1] * len(features_df),
                color='Type',
                title='Features Used in Clustering',
                color_discrete_map={'Numerical': '#003087', 'Categorical': '#1f77b4'}
            )
            fig.update_layout(showlegend=True, height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Model files
        st.markdown("## 📁 Model Artifacts")
        model_files = [
            ("Scaler", "models/scaler.pkl"),
            ("KMeans Model", "models/kmeans_model.pkl"), 
            ("GMM Model", "models/gmm_model.pkl"),
            ("Label Encoders", "models/label_encoders.pkl"),
            ("Metadata", "models/model_metadata.json")
        ]
        
        for name, file_path in model_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / 1024
                st.markdown(f'<div class="metric-card">✅ {name}: {file_size:.1f} KB</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="metric-card">❌ {name}: Missing</div>', unsafe_allow_html=True)
    
    def run(self):
        """Run the Streamlit application"""
        # Sidebar configuration
        st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=150)
        
        st.sidebar.title("🧭 Navigation")
        page = st.sidebar.radio(
            "Select Page",
            ["Customer Segmentation", "Analytics Dashboard", "Model Insights"],
            label_visibility="collapsed"
        )
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🏢 IBM Investment Analytics")
        st.sidebar.markdown("*Powered by Machine Learning*")
        
        # Route to appropriate page
        if page == "Customer Segmentation":
            self.customer_segmentation_page()
        elif page == "Analytics Dashboard":
            self.analytics_dashboard_page()
        elif page == "Model Insights":
            self.model_insights_page()
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")


def main():
    """Main application entry point"""
    app = InvestmentSegmentationApp()
    app.run()


if __name__ == "__main__":
    main()