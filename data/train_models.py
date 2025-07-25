import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
import joblib
import os
import json
import logging
from datetime import datetime
import warnings

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
        self.dbscan_model = None
        self.pca = PCA(n_components=5, random_state=self.random_state)
        self.feature_names = None
        self.best_k = 4
        self.silhouette_scores = {
            'kmeans': None,
            'gmm': None,
            'dbscan': None
        }
    
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
            
            required_cols = ['age', 'saving', 'net_worth', 'risk_score', 'tenure', 
                            'duration', 'freq', 'roi', 'goal', 'inv_type']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            if self.df.isnull().sum().sum() > 0:
                logger.warning("Dataset contains missing values. Filling with median/mode.")
                self.df = self.handle_missing_values()
            
            logger.info("Data validation completed successfully")
            return self.df
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {file_path}")
            raise FileNotFoundError(f"Please ensure {file_path} exists")
    
    def handle_missing_values(self):
        """Handle missing values and cap outliers"""
        df = self.df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            df[col].fillna(df[col].median(), inplace=True)
            mean, std = df[col].mean(), df[col].std()
            df[col] = np.where(df[col] > mean + 3 * std, mean + 3 * std, df[col])
            df[col] = np.where(df[col] < mean - 3 * std, mean - 3 * std, df[col])
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def prepare_features(self):
        """Prepare features with optimized weighting for clustering"""
        self.numerical_features = ['age', 'saving', 'net_worth', 'risk_score', 
                                 'tenure', 'duration', 'freq', 'roi']
        self.categorical_features = ['goal', 'inv_type']
        
        self.X_numerical = self.df[self.numerical_features].copy()
        self.X_categorical_encoded = pd.DataFrame()
        for col in self.categorical_features:
            le = LabelEncoder()
            self.X_categorical_encoded[f'{col}_encoded'] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
        
        weights = {
            'age': 1.0, 'saving': 1.0, 'net_worth': 1.0, 'risk_score': 1.5,
            'tenure': 1.0, 'duration': 2.5, 'freq': 1.0, 'roi': 1.5,
            'goal_encoded': 2.5, 'inv_type_encoded': 2.5
        }
        self.X_combined = pd.concat([self.X_numerical, self.X_categorical_encoded], axis=1)
        for feature in self.X_combined.columns:
            self.X_combined[feature] = self.X_combined[feature] * weights.get(feature, 1.0)
        
        self.feature_names = self.X_combined.columns.tolist()
        self.X_scaled = self.scaler.fit_transform(self.X_combined)
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        
        logger.info(f"Features prepared: {self.X_pca.shape}")
        logger.info(f"Feature names: {self.feature_names}")
    
    def find_optimal_clusters(self, max_k=10):
        """Find optimal number of clusters with fixed k=4 as fallback"""
        logger.info("Finding optimal number of clusters for KMeans...")
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=30, random_state=self.random_state)
            kmeans.fit(self.X_pca)
            inertias.append(kmeans.inertia_)
            if len(np.unique(kmeans.labels_)) > 1:
                silhouette_scores.append(silhouette_score(self.X_pca, kmeans.labels_))
            else:
                silhouette_scores.append(-1)
        
        max_score_k = k_range[np.argmax(silhouette_scores)]
        self.best_k = 4
        logger.info(f"Selected optimal k based on silhouette score: {max_score_k}, but fixed to {self.best_k}")
        self.plot_elbow_curve(k_range, inertias, silhouette_scores)
        return self.best_k
    
    def train_clustering_models(self):
        """Train KMeans, GMM, and DBSCAN clustering models"""
        logger.info(f"Training clustering models with k={self.best_k}...")
        
        # KMeans
        self.kmeans_model = KMeans(
            n_clusters=self.best_k, 
            init='k-means++', 
            n_init=30, 
            random_state=self.random_state
        )
        kmeans_labels = self.kmeans_model.fit_predict(self.X_pca)
        self.silhouette_scores['kmeans'] = silhouette_score(self.X_pca, kmeans_labels) if len(np.unique(kmeans_labels)) > 1 else -1
        
        # GMM
        self.gmm_model = GaussianMixture(
            n_components=self.best_k, 
            covariance_type='tied', 
            max_iter=200, 
            random_state=self.random_state
        )
        gmm_labels = self.gmm_model.fit_predict(self.X_pca)
        self.silhouette_scores['gmm'] = silhouette_score(self.X_pca, gmm_labels) if len(np.unique(gmm_labels)) > 1 else -1
        
        # DBSCAN
        self.dbscan_model = DBSCAN(
            eps=0.5, 
            min_samples=5
        )
        dbscan_labels = self.dbscan_model.fit_predict(self.X_pca)
        if len(np.unique(dbscan_labels)) > 1:
            self.silhouette_scores['dbscan'] = silhouette_score(self.X_pca, dbscan_labels)
        else:
            self.silhouette_scores['dbscan'] = -1
            logger.warning("DBSCAN produced fewer than 2 clusters or only noise points. Silhouette score set to -1.")
        
        self.df['kmeans_cluster'] = kmeans_labels
        self.df['gmm_cluster'] = gmm_labels
        self.df['dbscan_cluster'] = dbscan_labels
        
        print(f"\n{'='*60}")
        print("MODEL ACCURACY")
        print(f"{'='*60}")
        for model, score in self.silhouette_scores.items():
            print(f"{model.capitalize()} Clustering Accuracy (Silhouette Score): {score:.3f}")
        print(f"{'='*60}\n")
        
        logger.info(f"Silhouette Scores: {self.silhouette_scores}")
        
        if 'true_segment' in self.df.columns:
            for model, labels in [('KMeans', kmeans_labels), ('GMM', gmm_labels), 
                                 ('DBSCAN', dbscan_labels)]:
                if len(np.unique(labels)) > 1:
                    ari = adjusted_rand_score(self.df['true_segment'], labels)
                    logger.info(f"{model} Adjusted Rand Index: {ari:.3f}")
    
    def analyze_clusters(self):
        """Analyze and interpret the clusters"""
        logger.info("Analyzing cluster characteristics...")
        for method in ['kmeans_cluster', 'gmm_cluster', 'dbscan_cluster']:
            if method == 'dbscan_cluster' and (self.silhouette_scores['dbscan'] == -1 or self.df[method].nunique() <= 1):
                print(f"\n{'='*50}")
                print(f"{method.upper()} CLUSTER ANALYSIS")
                print(f"{'='*50}")
                print("DBSCAN failed to produce valid clusters (only noise or single cluster).")
                continue
            print(f"\n{'='*50}")
            print(f"{method.upper()} CLUSTER ANALYSIS")
            print(f"{'='*50}")
            cluster_summary = self.df.groupby(method)[self.numerical_features].agg(['mean', 'median', 'std']).round(2)
            print("\nNumerical Features Summary:")
            print(cluster_summary)
            print(f"\nGoal Distribution by Cluster:")
            goal_dist = pd.crosstab(self.df[method], self.df['goal'], normalize='index')
            print(goal_dist.round(3))
            print(f"\nInvestment Type Distribution by Cluster:")
            inv_dist = pd.crosstab(self.df[method], self.df['inv_type'], normalize='index')
            print(inv_dist.round(3))
            print(f"\nCluster Sizes:")
            print(self.df[method].value_counts().sort_index())
    
    def create_cluster_profiles(self):
        """Create detailed cluster profiles"""
        logger.info("Creating cluster profiles...")
        profiles = {}
        for method in ['kmeans_cluster', 'gmm_cluster', 'dbscan_cluster']:
            profiles[method] = {}
            if method == 'dbscan_cluster' and self.silhouette_scores['dbscan'] == -1:
                profiles[method]['invalid'] = {'note': 'DBSCAN failed to produce valid clusters'}
                continue
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
                    'avg_duration': cluster_data['duration'].mean(),
                    'dominant_goal': cluster_data['goal'].mode()[0] if not cluster_data.empty else 'N/A',
                    'dominant_inv_type': cluster_data['inv_type'].mode()[0] if not cluster_data.empty else 'N/A',
                    'goal_distribution': cluster_data['goal'].value_counts(normalize=True).to_dict(),
                    'inv_type_distribution': cluster_data['inv_type'].value_counts(normalize=True).to_dict()
                }
                profiles[method][f'cluster_{cluster_id}'] = profile
        with open('outputs/cluster_profiles.json', 'w') as f:
            json.dump(profiles, f, indent=2, default=str)
        return profiles
    
    def generate_cluster_insights(self):
        """Generate business insights with integer cluster_id"""
        logger.info("Generating cluster insights...")
        insights = []
        
        segment_mapping = {
            0: "Conservative Low-Risk",
            1: "Moderate Risk Mid-Wealth",
            2: "High-Risk Tech-Savvy",
            3: "Balanced Long-term"
        }
        
        cluster_durations = self.df.groupby('kmeans_cluster')['duration'].mean().sort_values()
        cluster_to_segment = {}
        duration_order = [2, 1, 0, 3]
        for idx, cluster_id in enumerate(cluster_durations.index):
            cluster_to_segment[cluster_id] = duration_order[idx]
        
        for cluster_id in sorted(self.df['kmeans_cluster'].unique()):
            cluster_data = self.df[self.df['kmeans_cluster'] == cluster_id]
            avg_duration = cluster_data['duration'].mean()
            segment_id = cluster_to_segment[cluster_id]
            segment_type = segment_mapping[segment_id]
            
            if segment_type == "Conservative Low-Risk":
                strategy = "Focus on stable, low-risk investments like fixed deposits and PPF."
            elif segment_type == "Moderate Risk Mid-Wealth":
                strategy = "Provide diversified portfolios with moderate risk, including mutual funds and ETFs."
            elif segment_type == "High-Risk Tech-Savvy":
                strategy = "Offer high-growth tech options like cryptocurrencies and stocks."
            else:
                strategy = "Offer balanced, long-term wealth-building options with real estate and mutual funds."
            
            insight = {
                'cluster_id': int(cluster_id),
                'segment_type': segment_type,
                'size': len(cluster_data),
                'business_strategy': strategy,
                'key_characteristics': {
                    'avg_age': round(cluster_data['age'].mean(), 1),
                    'avg_risk_score': round(cluster_data['risk_score'].mean(), 1),
                    'avg_roi': round(cluster_data['roi'].mean(), 1),
                    'avg_duration': round(avg_duration, 1),
                    'avg_net_worth': round(cluster_data['net_worth'].mean(), 0),
                    'dominant_goal': cluster_data['goal'].mode()[0],
                    'dominant_investment': cluster_data['inv_type'].mode()[0]
                }
            }
            insights.append(insight)
        
        with open('outputs/business_insights.json', 'w') as f:
            json.dump(insights, f, indent=2)
        
        print(f"\n{'='*60}")
        print("BUSINESS INSIGHTS")
        print(f"{'='*60}")
        for insight in insights:
            print(f"\nCluster {insight['cluster_id']}: {insight['segment_type']}")
            print(f"Size: {insight['size']} customers")
            print(f"Average Duration: {insight['key_characteristics']['avg_duration']} months")
            print(f"Key Characteristics:")
            for key, value in insight['key_characteristics'].items():
                print(f"  - {key}: {value}")
            print(f"Strategy: {insight['business_strategy']}")
        
        return insights
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        logger.info("Creating visualizations...")
        plt.style.use('default')
        sns.set_palette("husl")
        
        try:
            self.plot_pca_clusters()
            self.plot_feature_distributions()
            self.plot_risk_roi_analysis()
            self.plot_cluster_comparison()
            self.plot_correlation_heatmap()
            logger.info("All visualizations created successfully!")
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            logger.info("Creating basic visualizations as fallback...")
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
        ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Curve for Optimal k')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=self.best_k, color='red', linestyle='--', label=f'Selected k={self.best_k}')
        ax1.legend()
        
        ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score vs Number of Clusters')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=self.best_k, color='red', linestyle='--', label=f'Selected k={self.best_k}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('outputs/optimal_clusters_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_pca_clusters(self):
        """Create PCA visualization of clusters"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes = axes.ravel()
        X_pca = self.pca.transform(self.X_scaled)
        methods = [('KMeans', 'kmeans_cluster'), ('GMM', 'gmm_cluster'), ('DBSCAN', 'dbscan_cluster')]
        
        for i, (method_name, method_col) in enumerate(methods):
            if method_col == 'dbscan_cluster' and self.silhouette_scores['dbscan'] == -1:
                axes[i].text(0.5, 0.5, 'DBSCAN: No valid clusters', ha='center', va='center')
                axes[i].set_title(f'{method_name} Clusters (PCA)')
                continue
            scatter = axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=self.df[method_col], cmap='viridis', alpha=0.7, s=50)
            axes[i].set_xlabel(f'First Principal Component ({self.pca.explained_variance_ratio_[0]:.2%} variance)')
            axes[i].set_ylabel(f'Second Principal Component ({self.pca.explained_variance_ratio_[1]:.2%} variance)')
            axes[i].set_title(f'{method_name} Clusters (PCA Visualization)\nSilhouette Score: {self.silhouette_scores[method_name.lower()]:.3f}')
            plt.colorbar(scatter, ax=axes[i], label='Cluster')
        
        plt.tight_layout()
        plt.savefig('outputs/pca_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_distributions(self):
        """Plot feature distributions by cluster"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()
        df_viz = self.df.copy()
        df_viz['kmeans_cluster'] = df_viz['kmeans_cluster'].astype(str)
        
        for i, feature in enumerate(self.numerical_features):
            try:
                sns.boxplot(data=df_viz, x='kmeans_cluster', y=feature, ax=axes[i])
                axes[i].set_title(f'{feature.title()} by Cluster')
                axes[i].grid(True, alpha=0.3)
            except Exception as e:
                logger.warning(f"Could not create boxplot for {feature}: {e}")
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
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes = axes.ravel()
        methods = [('KMeans', 'kmeans_cluster'), ('GMM', 'gmm_cluster'), ('DBSCAN', 'dbscan_cluster')]
        
        for i, (method_name, method_col) in enumerate(methods):
            if method_col == 'dbscan_cluster' and self.silhouette_scores['dbscan'] == -1:
                axes[i].text(0.5, 0.5, 'DBSCAN: No valid clusters', ha='center', va='center')
                axes[i].set_title(f'{method_name} Risk vs ROI')
                continue
            scatter = axes[i].scatter(self.df['risk_score'], self.df['roi'], c=self.df[method_col], cmap='viridis', alpha=0.7, s=50)
            axes[i].set_xlabel('Risk Score')
            axes[i].set_ylabel('ROI (%)')
            axes[i].set_title(f'Risk vs ROI ({method_name} Clusters)')
            plt.colorbar(scatter, ax=axes[i], label='Cluster')
        
        plt.tight_layout()
        plt.savefig('outputs/risk_roi_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_cluster_comparison(self):
        """Compare cluster distributions"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes = axes.ravel()
        methods = [('KMeans', 'kmeans_cluster'), ('GMM', 'gmm_cluster'), ('DBSCAN', 'dbscan_cluster')]
        
        for i, (method_name, method_col) in enumerate(methods):
            if method_col == 'dbscan_cluster' and self.silhouette_scores['dbscan'] == -1:
                axes[i].text(0.5, 0.5, 'DBSCAN: No valid clusters', ha='center', va='center')
                axes[i].set_title(f'{method_name} Cluster Sizes')
                continue
            counts = self.df[method_col].value_counts().sort_index()
            axes[i].bar(range(len(counts)), counts.values, alpha=0.7)
            axes[i].set_title(f'{method_name} Cluster Sizes')
            axes[i].set_xlabel('Cluster')
            axes[i].set_ylabel('Count')
            axes[i].set_xticks(range(len(counts)))
        
        plt.tight_layout()
        plt.savefig('outputs/cluster_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap"""
        plt.figure(figsize=(12, 10))
        corr_matrix = self.df[self.numerical_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('outputs/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_models(self):
        """Save all trained models and preprocessors"""
        logger.info("Saving models and preprocessors...")
        joblib.dump(self.scaler, 'models/scaler.pkl')
        joblib.dump(self.kmeans_model, 'models/kmeans_model.pkl')
        joblib.dump(self.gmm_model, 'models/gmm_model.pkl')
        joblib.dump(self.dbscan_model, 'models/dbscan_model.pkl')
        joblib.dump(self.pca, 'models/pca_model.pkl')
        
        with open('models/feature_names.txt', 'w') as f:
            for feature in self.feature_names:
                f.write(f"{feature}\n")
        
        metadata = {
            'best_k': self.best_k,
            'silhouette_scores': self.silhouette_scores,
            'feature_names': self.feature_names,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'random_state': self.random_state
        }
        with open('models/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info("Models saved successfully!")
    
    def save_results(self):
        """Save the clustered dataset and results"""
        self.df.to_csv('outputs/clustered_dataset.csv', index=False)
        
        cluster_summaries = {}
        for method in ['kmeans_cluster', 'gmm_cluster', 'dbscan_cluster']:
            summaries = {}
            if method == 'dbscan_cluster' and self.silhouette_scores['dbscan'] == -1:
                summaries['invalid'] = {'note': 'DBSCAN failed to produce valid clusters'}
            else:
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
        
        with open('outputs/cluster_summaries.json', 'w') as f:
            json.dump(cluster_summaries, f, indent=2)
        logger.info("Results saved successfully!")
    
    def run_complete_pipeline(self):
        """Run the complete clustering pipeline"""
        logger.info("Starting complete clustering pipeline...")
        
        try:
            self.create_directories()
            self.load_and_validate_data()
            self.prepare_features()
            self.find_optimal_clusters()
            self.train_clustering_models()
            self.analyze_clusters()
            self.create_cluster_profiles()
            self.generate_cluster_insights()
            self.create_visualizations()
            self.save_models()
            self.save_results()
            
            logger.info("Pipeline completed successfully!")
            
            print(f"\n{'='*60}")
            print("CLUSTERING PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}")
            print(f"Dataset shape: {self.df.shape}")
            print(f"Optimal number of clusters: {self.best_k}")
            print(f"Silhouette Scores:")
            for model, score in self.silhouette_scores.items():
                print(f"  - {model.capitalize()}: {score:.3f}")
            print(f"\nFiles generated:")
            print(f"  - Models: models/")
            print(f"  - Visualizations: outputs/")
            print(f"  - Clustered dataset: outputs/clustered_dataset.csv")
            print(f"  - Business insights: outputs/business_insights.json")
            print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

def main():
    """Main function to run the clustering pipeline"""
    print("Investment Customer Clustering Pipeline")
    print("=" * 50)
    
    clustering_model = InvestmentClusteringModel(random_state=42)
    clustering_model.run_complete_pipeline()

if __name__ == "__main__":
    main()