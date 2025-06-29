import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InvestmentClusteringModel:
    """
    A comprehensive clustering model for investment customer segmentation
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.kmeans_model = None
        self.gmm_model = None
        self.pca = None
        self.feature_names = None
        self.best_k = 4  # Default, will be determined by elbow method
        
    def create_directories(self):
        """Create necessary output directories"""
        directories = ['models', 'outputs', 'data']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created/verified directory: {directory}")
    
    def load_and_validate_data(self, file_path="data/investment_customers.csv"):
        """Load and validate the dataset"""
        try:
            self.df = pd.read_csv(file_path)
            logger.info(f"Dataset loaded successfully: {self.df.shape}")
            
            # Required columns for clustering
            required_cols = ['age', 'saving', 'net_worth', 'risk_score', 'tenure', 
                           'duration', 'freq', 'roi', 'goal', 'inv_type']
            
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Check for missing values
            if self.df.isnull().sum().sum() > 0:
                logger.warning("Dataset contains missing values. Filling with median/mode.")
                self.df = self.handle_missing_values()
            
            logger.info("Data validation completed successfully")
            return self.df
            
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {file_path}")
            raise FileNotFoundError(f"Please ensure {file_path} exists")
    
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        df = self.df.copy()
        
        # Fill numerical columns with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def prepare_features(self):
        """Prepare features for clustering"""
        # Define feature sets
        self.numerical_features = ['age', 'saving', 'net_worth', 'risk_score', 
                                 'tenure', 'duration', 'freq', 'roi']
        self.categorical_features = ['goal', 'inv_type']
        
        # Extract numerical features
        self.X_numerical = self.df[self.numerical_features].copy()
        
        # Encode categorical features
        self.X_categorical_encoded = pd.DataFrame()
        for col in self.categorical_features:
            le = LabelEncoder()
            self.X_categorical_encoded[f'{col}_encoded'] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
        
        # Combine all features
        self.X_combined = pd.concat([self.X_numerical, self.X_categorical_encoded], axis=1)
        self.feature_names = self.X_combined.columns.tolist()
        
        # Scale the features
        self.X_scaled = self.scaler.fit_transform(self.X_combined)
        
        logger.info(f"Features prepared: {self.X_scaled.shape}")
        logger.info(f"Feature names: {self.feature_names}")
    
    def find_optimal_clusters(self, max_k=10):
        """Find optimal number of clusters using elbow method and silhouette analysis"""
        logger.info("Finding optimal number of clusters...")
        
        # Elbow method
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=20, 
                          random_state=self.random_state)
            kmeans.fit(self.X_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.X_scaled, kmeans.labels_))
        
        # Find elbow point (simple method - look for the "knee")
        # Calculate the rate of change
        rate_of_change = []
        for i in range(1, len(inertias)):
            rate_of_change.append(inertias[i-1] - inertias[i])
        
        # Find the point where rate of change starts to level off
        optimal_k_elbow = k_range[np.argmax(rate_of_change[1:]) + 2]  # +2 for indexing adjustment
        
        # Find optimal k based on silhouette score
        optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
        
        # Choose the better option (prefer silhouette if reasonable)
        if abs(optimal_k_elbow - optimal_k_silhouette) <= 1:
            self.best_k = optimal_k_silhouette
        else:
            self.best_k = optimal_k_elbow
        
        logger.info(f"Optimal k by elbow method: {optimal_k_elbow}")
        logger.info(f"Optimal k by silhouette score: {optimal_k_silhouette}")
        logger.info(f"Selected optimal k: {self.best_k}")
        
        # Plot elbow curve
        self.plot_elbow_curve(k_range, inertias, silhouette_scores)
        
        return self.best_k
    
    def train_clustering_models(self):
        """Train both KMeans and GMM clustering models"""
        logger.info(f"Training clustering models with k={self.best_k}...")
        
        # Train KMeans
        self.kmeans_model = KMeans(
            n_clusters=self.best_k, 
            init='k-means++', 
            n_init=20, 
            random_state=self.random_state
        )
        kmeans_labels = self.kmeans_model.fit_predict(self.X_scaled)
        
        # Train Gaussian Mixture Model
        self.gmm_model = GaussianMixture(
            n_components=self.best_k, 
            covariance_type='full', 
            random_state=self.random_state
        )
        gmm_labels = self.gmm_model.fit_predict(self.X_scaled)
        
        # Add cluster labels to dataframe
        self.df['kmeans_cluster'] = kmeans_labels
        self.df['gmm_cluster'] = gmm_labels
        
        # Calculate cluster quality metrics
        self.kmeans_silhouette = silhouette_score(self.X_scaled, kmeans_labels)
        self.gmm_silhouette = silhouette_score(self.X_scaled, gmm_labels)
        
        logger.info(f"KMeans Silhouette Score: {self.kmeans_silhouette:.3f}")
        logger.info(f"GMM Silhouette Score: {self.gmm_silhouette:.3f}")
        
        # If true labels exist, calculate ARI
        if 'true_segment' in self.df.columns:
            kmeans_ari = adjusted_rand_score(self.df['true_segment'], kmeans_labels)
            gmm_ari = adjusted_rand_score(self.df['true_segment'], gmm_labels)
            logger.info(f"KMeans Adjusted Rand Index: {kmeans_ari:.3f}")
            logger.info(f"GMM Adjusted Rand Index: {gmm_ari:.3f}")
    
    def analyze_clusters(self):
        """Analyze and interpret the clusters"""
        logger.info("Analyzing cluster characteristics...")
        
        # Cluster summaries for both methods
        for method in ['kmeans_cluster', 'gmm_cluster']:
            print(f"\n{'='*50}")
            print(f"{method.upper()} CLUSTER ANALYSIS")
            print(f"{'='*50}")
            
            # Numerical features summary
            cluster_summary = self.df.groupby(method)[self.numerical_features].agg([
                'mean', 'median', 'std'
            ]).round(2)
            
            print("\nNumerical Features Summary:")
            print(cluster_summary)
            
            # Categorical features distribution
            print(f"\nGoal Distribution by Cluster:")
            goal_dist = pd.crosstab(self.df[method], self.df['goal'], normalize='index')
            print(goal_dist.round(3))
            
            print(f"\nInvestment Type Distribution by Cluster:")
            inv_dist = pd.crosstab(self.df[method], self.df['inv_type'], normalize='index')
            print(inv_dist.round(3))
            
            # Cluster size
            print(f"\nCluster Sizes:")
            print(self.df[method].value_counts().sort_index())
    
    def create_cluster_profiles(self):
        """Create detailed cluster profiles with interpretations"""
        logger.info("Creating cluster profiles...")
        
        profiles = {}
        
        for method in ['kmeans_cluster', 'gmm_cluster']:
            profiles[method] = {}
            
            for cluster_id in sorted(self.df[method].unique()):
                cluster_data = self.df[self.df[method] == cluster_id]
                
                profile = {
                    'size': len(cluster_data),
                    'percentage': len(cluster_data) / len(self.df) * 100,
                    'avg_age': cluster_data['age'].mean(),
                    'avg_saving': cluster_data['saving'].mean(),
                    'avg_net_worth': cluster_data['net_worth'].mean(),
                    'avg_risk_score': cluster_data['risk_score'].mean(),
                    'avg_roi': cluster_data['roi'].mean(),
                    'avg_tenure': cluster_data['tenure'].mean(),
                    'dominant_goal': cluster_data['goal'].mode()[0],
                    'dominant_inv_type': cluster_data['inv_type'].mode()[0],
                    'goal_distribution': cluster_data['goal'].value_counts(normalize=True).to_dict(),
                    'inv_type_distribution': cluster_data['inv_type'].value_counts(normalize=True).to_dict()
                }
                
                profiles[method][f'cluster_{cluster_id}'] = profile
        
        # Save profiles
        import json
        with open('outputs/cluster_profiles.json', 'w') as f:
            json.dump(profiles, f, indent=2, default=str)
        
        return profiles
    
    def generate_cluster_insights(self):
        """Generate business insights from clusters"""
        logger.info("Generating cluster insights...")
        
        insights = []
        
        # Use KMeans clusters for insights (you can change this)
        for cluster_id in sorted(self.df['kmeans_cluster'].unique()):
            cluster_data = self.df[self.df['kmeans_cluster'] == cluster_id]
            
            avg_age = cluster_data['age'].mean()
            avg_risk = cluster_data['risk_score'].mean()
            avg_roi = cluster_data['roi'].mean()
            avg_net_worth = cluster_data['net_worth'].mean()
            dominant_goal = cluster_data['goal'].mode()[0]
            dominant_inv = cluster_data['inv_type'].mode()[0]
            
            # Generate insight based on characteristics
            if avg_age > 45 and avg_risk < 30:
                segment_type = "Conservative Seniors"
                strategy = "Focus on stable, low-risk investments with guaranteed returns"
            elif avg_age < 35 and avg_risk > 50:
                segment_type = "Aggressive Young Investors"
                strategy = "Offer high-growth, tech-focused investment options"
            elif avg_net_worth > 800000:
                segment_type = "High Net Worth Individuals"
                strategy = "Provide premium investment services and wealth management"
            else:
                segment_type = "Balanced Investors"
                strategy = "Offer diversified portfolio options"
            
            insight = {
                'cluster_id': cluster_id,
                'segment_type': segment_type,
                'size': len(cluster_data),
                'key_characteristics': {
                    'avg_age': round(avg_age, 1),
                    'avg_risk_score': round(avg_risk, 1),
                    'avg_roi': round(avg_roi, 1),
                    'avg_net_worth': round(avg_net_worth, 0),
                    'dominant_goal': dominant_goal,
                    'dominant_investment': dominant_inv
                },
                'business_strategy': strategy
            }
            
            insights.append(insight)
        
        # Save insights
        import json
        with open('outputs/business_insights.json', 'w') as f:
            json.dump(insights, f, indent=2, default=str)
        
        # Print insights
        print(f"\n{'='*60}")
        print("BUSINESS INSIGHTS")
        print(f"{'='*60}")
        
        for insight in insights:
            print(f"\nCluster {insight['cluster_id']}: {insight['segment_type']}")
            print(f"Size: {insight['size']} customers")
            print(f"Key Characteristics:")
            for key, value in insight['key_characteristics'].items():
                print(f"  - {key}: {value}")
            print(f"Strategy: {insight['business_strategy']}")
        
        return insights
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        logger.info("Creating visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        try:
            # 1. PCA visualization
            self.plot_pca_clusters()
            
            # 2. Feature distributions by cluster
            self.plot_feature_distributions()
            
            # 3. Risk vs ROI analysis
            self.plot_risk_roi_analysis()
            
            # 4. Cluster comparison
            self.plot_cluster_comparison()
            
            # 5. Correlation heatmap
            self.plot_correlation_heatmap()
            
            logger.info("All visualizations created successfully!")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            logger.info("Creating basic visualizations as fallback...")
            
            # Create a simple summary plot as fallback
            fig, ax = plt.subplots(figsize=(10, 6))
            self.df['kmeans_cluster'].value_counts().sort_index().plot(kind='bar', ax=ax)
            ax.set_title('Cluster Distribution (KMeans)')
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Count')
            plt.tight_layout()
            plt.savefig('outputs/basic_cluster_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_elbow_curve(self, k_range, inertias, silhouette_scores):
        """Plot elbow curve and silhouette scores"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Elbow curve
        ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Curve for Optimal k')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=self.best_k, color='red', linestyle='--', 
                   label=f'Selected k={self.best_k}')
        ax1.legend()
        
        # Silhouette scores
        ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score vs Number of Clusters')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=self.best_k, color='red', linestyle='--', 
                   label=f'Selected k={self.best_k}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('outputs/optimal_clusters_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_pca_clusters(self):
        """Create PCA visualization of clusters"""
        # Perform PCA
        self.pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = self.pca.fit_transform(self.X_scaled)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # KMeans clusters
        scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], 
                             c=self.df['kmeans_cluster'], 
                             cmap='viridis', alpha=0.7, s=50)
        ax1.set_xlabel(f'First Principal Component ({self.pca.explained_variance_ratio_[0]:.2%} variance)')
        ax1.set_ylabel(f'Second Principal Component ({self.pca.explained_variance_ratio_[1]:.2%} variance)')
        ax1.set_title(f'KMeans Clusters (PCA Visualization)\nSilhouette Score: {self.kmeans_silhouette:.3f}')
        plt.colorbar(scatter1, ax=ax1, label='Cluster')
        
        # GMM clusters
        scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], 
                             c=self.df['gmm_cluster'], 
                             cmap='viridis', alpha=0.7, s=50)
        ax2.set_xlabel(f'First Principal Component ({self.pca.explained_variance_ratio_[0]:.2%} variance)')
        ax2.set_ylabel(f'Second Principal Component ({self.pca.explained_variance_ratio_[1]:.2%} variance)')
        ax2.set_title(f'GMM Clusters (PCA Visualization)\nSilhouette Score: {self.gmm_silhouette:.3f}')
        plt.colorbar(scatter2, ax=ax2, label='Cluster')
        
        plt.tight_layout()
        plt.savefig('outputs/pca_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_distributions(self):
        """Plot feature distributions by cluster"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()
        
        # Convert cluster labels to string to avoid seaborn issues
        df_viz = self.df.copy()
        df_viz['kmeans_cluster'] = df_viz['kmeans_cluster'].astype(str)
        
        for i, feature in enumerate(self.numerical_features):
            try:
                sns.boxplot(data=df_viz, x='kmeans_cluster', y=feature, ax=axes[i])
                axes[i].set_title(f'{feature.title()} by Cluster')
                axes[i].grid(True, alpha=0.3)
            except Exception as e:
                logger.warning(f"Could not create boxplot for {feature}: {e}")
                # Fallback to histogram
                for cluster in df_viz['kmeans_cluster'].unique():
                    cluster_data = df_viz[df_viz['kmeans_cluster'] == cluster][feature]
                    axes[i].hist(cluster_data, alpha=0.7, label=f'Cluster {cluster}')
                axes[i].set_title(f'{feature.title()} by Cluster (Histogram)')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_risk_roi_analysis(self):
        """Plot risk vs ROI analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # KMeans
        scatter1 = ax1.scatter(self.df['risk_score'], self.df['roi'], 
                             c=self.df['kmeans_cluster'], 
                             cmap='viridis', alpha=0.7, s=50)
        ax1.set_xlabel('Risk Score')
        ax1.set_ylabel('ROI (%)')
        ax1.set_title('Risk vs ROI (KMeans Clusters)')
        plt.colorbar(scatter1, ax=ax1, label='Cluster')
        
        # GMM
        scatter2 = ax2.scatter(self.df['risk_score'], self.df['roi'], 
                             c=self.df['gmm_cluster'], 
                             cmap='viridis', alpha=0.7, s=50)
        ax2.set_xlabel('Risk Score')
        ax2.set_ylabel('ROI (%)')
        ax2.set_title('Risk vs ROI (GMM Clusters)')
        plt.colorbar(scatter2, ax=ax2, label='Cluster')
        
        plt.tight_layout()
        plt.savefig('outputs/risk_roi_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_cluster_comparison(self):
        """Compare cluster distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Cluster sizes
        kmeans_counts = self.df['kmeans_cluster'].value_counts().sort_index()
        gmm_counts = self.df['gmm_cluster'].value_counts().sort_index()
        
        axes[0, 0].bar(range(len(kmeans_counts)), kmeans_counts.values, alpha=0.7)
        axes[0, 0].set_title('KMeans Cluster Sizes')
        axes[0, 0].set_xlabel('Cluster')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_xticks(range(len(kmeans_counts)))
        
        axes[0, 1].bar(range(len(gmm_counts)), gmm_counts.values, alpha=0.7)
        axes[0, 1].set_title('GMM Cluster Sizes')
        axes[0, 1].set_xlabel('Cluster')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_xticks(range(len(gmm_counts)))
        
        # Goal distributions with error handling
        try:
            goal_kmeans = pd.crosstab(self.df['kmeans_cluster'].astype(str), 
                                    self.df['goal'], normalize='index')
            goal_gmm = pd.crosstab(self.df['gmm_cluster'].astype(str), 
                                 self.df['goal'], normalize='index')
            
            goal_kmeans.plot(kind='bar', ax=axes[1, 0], stacked=True)
            axes[1, 0].set_title('Goal Distribution by KMeans Cluster')
            axes[1, 0].set_xlabel('Cluster')
            axes[1, 0].set_ylabel('Proportion')
            axes[1, 0].legend(title='Goal', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            goal_gmm.plot(kind='bar', ax=axes[1, 1], stacked=True)
            axes[1, 1].set_title('Goal Distribution by GMM Cluster')
            axes[1, 1].set_xlabel('Cluster')
            axes[1, 1].set_ylabel('Proportion')
            axes[1, 1].legend(title='Goal', bbox_to_anchor=(1.05, 1), loc='upper left')
            
        except Exception as e:
            logger.warning(f"Could not create goal distribution plots: {e}")
            # Create simple bar plots instead
            axes[1, 0].bar(range(len(kmeans_counts)), kmeans_counts.values, alpha=0.7)
            axes[1, 0].set_title('KMeans Cluster Sizes (Fallback)')
            axes[1, 1].bar(range(len(gmm_counts)), gmm_counts.values, alpha=0.7)
            axes[1, 1].set_title('GMM Cluster Sizes (Fallback)')
        
        plt.tight_layout()
        plt.savefig('outputs/cluster_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap"""
        plt.figure(figsize=(12, 10))
        
        # Calculate correlation matrix
        corr_matrix = self.df[self.numerical_features].corr()
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('outputs/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_models(self):
        """Save all trained models and preprocessors"""
        logger.info("Saving models and preprocessors...")
        
        # Save models
        joblib.dump(self.scaler, 'models/scaler.pkl')
        joblib.dump(self.kmeans_model, 'models/kmeans_model.pkl')
        joblib.dump(self.gmm_model, 'models/gmm_model.pkl')
        joblib.dump(self.label_encoders, 'models/label_encoders.pkl')
        
        if self.pca:
            joblib.dump(self.pca, 'models/pca_model.pkl')
        
        # Save feature names
        with open('models/feature_names.txt', 'w') as f:
            for feature in self.feature_names:
                f.write(f"{feature}\n")
        
        # Save model metadata
        metadata = {
            'best_k': self.best_k,
            'kmeans_silhouette': self.kmeans_silhouette,
            'gmm_silhouette': self.gmm_silhouette,
            'feature_names': self.feature_names,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'random_state': self.random_state
        }
        
        import json
        with open('models/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info("Models saved successfully!")
    
    def save_results(self):
        """Save the clustered dataset and results"""
        # Save the dataset with cluster labels
        self.df.to_csv('outputs/clustered_dataset.csv', index=False)
        
        # Save cluster summaries
        cluster_summaries = {}
        
        for method in ['kmeans_cluster', 'gmm_cluster']:
            summaries = {}
            for cluster_id in sorted(self.df[method].unique()):
                cluster_data = self.df[self.df[method] == cluster_id]
                
                summary = {
                    'size': len(cluster_data),
                    'percentage': len(cluster_data) / len(self.df) * 100,
                    'numerical_stats': cluster_data[self.numerical_features].describe().to_dict(),
                    'goal_distribution': cluster_data['goal'].value_counts().to_dict(),
                    'inv_type_distribution': cluster_data['inv_type'].value_counts().to_dict()
                }
                
                summaries[f'cluster_{cluster_id}'] = summary
            
            cluster_summaries[method] = summaries
        
        import json
        with open('outputs/cluster_summaries.json', 'w') as f:
            json.dump(cluster_summaries, f, indent=2, default=str)
        
        logger.info("Results saved successfully!")
    
    def run_complete_pipeline(self):
        """Run the complete clustering pipeline"""
        logger.info("Starting complete clustering pipeline...")
        
        try:
            # Step 1: Setup
            self.create_directories()
            
            # Step 2: Load and validate data
            self.load_and_validate_data()
            
            # Step 3: Prepare features
            self.prepare_features()
            
            # Step 4: Find optimal number of clusters
            self.find_optimal_clusters()
            
            # Step 5: Train clustering models
            self.train_clustering_models()
            
            # Step 6: Analyze clusters
            self.analyze_clusters()
            
            # Step 7: Create cluster profiles
            self.create_cluster_profiles()
            
            # Step 8: Generate business insights
            self.generate_cluster_insights()
            
            # Step 9: Create visualizations
            self.create_visualizations()
            
            # Step 10: Save models and results
            self.save_models()
            self.save_results()
            
            logger.info("Pipeline completed successfully!")
            
            # Print final summary
            print(f"\n{'='*60}")
            print("CLUSTERING PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}")
            print(f"Dataset shape: {self.df.shape}")
            print(f"Optimal number of clusters: {self.best_k}")
            print(f"KMeans Silhouette Score: {self.kmeans_silhouette:.3f}")
            print(f"GMM Silhouette Score: {self.gmm_silhouette:.3f}")
            print(f"\nFiles generated:")
            print(f"  - Models: models/")
            print(f"  - Visualizations: outputs/")
            print(f"  - Clustered dataset: outputs/clustered_dataset.csv")
            print(f"  - Business insights: outputs/business_insights.json")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    """Main function to run the clustering pipeline"""
    print("Investment Customer Clustering Pipeline")
    print("=" * 50)
    
    # Initialize the clustering model
    clustering_model = InvestmentClusteringModel(random_state=42)
    
    # Run the complete pipeline
    clustering_model.run_complete_pipeline()


if __name__ == "__main__":
    main()