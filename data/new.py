import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score

np.random.seed(42)
n_customers = 5000  # Increased for robustness

def generate_customer_segments():
    """
    Generate 4 customer segments with distinct distributions for better clustering.
    """
    # Segment 0: Conservative Low-Risk Investors (25%)
    n_conservative = int(0.25 * n_customers)
    conservative_data = {
        'age': np.random.normal(65, 5, n_conservative).astype(int),
        'saving': np.random.normal(10000, 2000, n_conservative),
        'net_worth': np.random.normal(6000000, 500000, n_conservative),  # ~60L
        'tenure': np.random.normal(15, 2, n_conservative),
        'duration': np.random.normal(80, 5, n_conservative),  # Long duration
        'freq': np.random.normal(2, 0.5, n_conservative),
        'goal': np.random.choice(['retirement', 'emergency'], n_conservative, p=[0.9, 0.1]),
        'inv_type': np.random.choice(['fd', 'ppf', 'gold'], n_conservative, p=[0.7, 0.2, 0.1])
    }

    # Segment 1: Moderate Risk Mid-Wealth (30%)
    n_moderate = int(0.30 * n_customers)
    moderate_data = {
        'age': np.random.normal(40, 5, n_moderate).astype(int),
        'saving': np.random.normal(25000, 3000, n_moderate),
        'net_worth': np.random.normal(2000000, 400000, n_moderate),  # ~20L
        'tenure': np.random.normal(8, 1.5, n_moderate),
        'duration': np.random.normal(50, 5, n_moderate),  # Medium duration
        'freq': np.random.normal(10, 2, n_moderate),
        'goal': np.random.choice(['education', 'wealth'], n_moderate, p=[0.8, 0.2]),
        'inv_type': np.random.choice(['mutual', 'etf', 'stocks'], n_moderate, p=[0.6, 0.3, 0.1])
    }

    # Segment 2: High-Risk Tech-Savvy (25%)
    n_aggressive = int(0.25 * n_customers)
    aggressive_data = {
        'age': np.random.normal(25, 4, n_aggressive).astype(int),
        'saving': np.random.normal(35000, 4000, n_aggressive),
        'net_worth': np.random.normal(1000000, 300000, n_aggressive),  # ~10L
        'tenure': np.random.normal(2, 1, n_aggressive),
        'duration': np.random.normal(12, 3, n_aggressive),  # Short duration
        'freq': np.random.normal(20, 3, n_aggressive),
        'goal': np.random.choice(['wealth', 'vacation'], n_aggressive, p=[0.9, 0.1]),
        'inv_type': np.random.choice(['crypto', 'stocks'], n_aggressive, p=[0.7, 0.3])
    }

    # Segment 3: Balanced Long-term Wealth Builders (20%)
    n_balanced = n_customers - n_conservative - n_moderate - n_aggressive
    balanced_data = {
        'age': np.random.normal(50, 5, n_balanced).astype(int),
        'saving': np.random.normal(40000, 3500, n_balanced),
        'net_worth': np.random.normal(3500000, 600000, n_balanced),  # ~35L
        'tenure': np.random.normal(12, 2, n_balanced),
        'duration': np.random.normal(100, 5, n_balanced),  # Very long duration
        'freq': np.random.normal(6, 1.5, n_balanced),
        'goal': np.random.choice(['retirement', 'wealth'], n_balanced, p=[0.8, 0.2]),
        'inv_type': np.random.choice(['real_estate', 'mutual', 'etf'], n_balanced, p=[0.6, 0.3, 0.1])
    }

    # Combine segments
    all_data = {}
    for key in conservative_data.keys():
        all_data[key] = np.concatenate([
            conservative_data[key],
            moderate_data[key],
            aggressive_data[key],
            balanced_data[key]
        ])

    # True labels
    true_labels = np.concatenate([
        np.zeros(n_conservative, dtype=int),
        np.ones(n_moderate, dtype=int),
        np.full(n_aggressive, 2, dtype=int),
        np.full(n_balanced, 3, dtype=int)
    ])

    return all_data, true_labels

def clean_and_validate_data(data):
    """Clean and validate data with realistic constraints"""
    data['age'] = np.clip(data['age'], 18, 75)
    data['saving'] = np.clip(data['saving'], 2000, 100000)
    data['net_worth'] = np.clip(data['net_worth'], 50000, 10000000)
    data['tenure'] = np.clip(data['tenure'], 0, 30).astype(int)
    data['duration'] = np.clip(data['duration'], 1, 120).astype(int)
    data['freq'] = np.clip(data['freq'], 1, 30).astype(int)
    return data

def calculate_risk_score(age, saving, freq, duration):
    """Calculate risk score with adjusted weights for higher, realistic values"""
    score = (age * 0.5) + (saving / 5000) + (freq * 2.0) - (duration * 0.1)
    return np.clip(score, 10, 80).round(1)  # Adjusted clip range for better differentiation

def calculate_roi(risk_score, freq, inv_type):
    """Calculate ROI with higher, realistic values"""
    base_roi = 6  # Increased base for realism
    risk_bonus = risk_score / 10  # Increased impact of risk_score
    freq_bonus = freq * 0.2  # Increased frequency impact
    inv_type_bonus_map = {
        'stocks': 7, 'mutual': 5, 'crypto': 12, 'gold': 3,
        'fd': 2, 'ppf': 2.5, 'real_estate': 5, 'etf': 5
    }
    inv_type_bonus = np.array([inv_type_bonus_map.get(inv, 3) for inv in inv_type])
    roi = base_roi + risk_bonus + freq_bonus + inv_type_bonus
    return np.clip(roi, 6, 30).round(2)  # Clip to realistic ROI range

def create_visualization(df, true_labels):
    """Create visualizations and compute silhouette score"""
    os.makedirs('outputs', exist_ok=True)
    plt.style.use('default')
    sns.set_palette("husl")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Optimized Investment Customer Dataset - Segment Analysis', fontsize=16)

    # Age distribution
    for i in range(4):
        segment_data = df[true_labels == i]
        axes[0, 0].hist(segment_data['age'], alpha=0.7, label=f'Segment {i}', bins=20)
    axes[0, 0].set_title('Age Distribution by Segment')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()

    # Risk score distribution
    for i in range(4):
        segment_data = df[true_labels == i]
        axes[0, 1].hist(segment_data['risk_score'], alpha=0.7, label=f'Segment {i}', bins=20)
    axes[0, 1].set_title('Risk Score Distribution by Segment')
    axes[0, 1].set_xlabel('Risk Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()

    # Net worth distribution
    for i in range(4):
        segment_data = df[true_labels == i]
        axes[0, 2].hist(segment_data['net_worth'] / 100000, alpha=0.7, label=f'Segment {i}', bins=20)
    axes[0, 2].set_title('Net Worth Distribution (₹ in lakhs)')
    axes[0, 2].set_xlabel('Net Worth (₹ lakhs)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()

    # Investment type distribution
    inv_type_counts = df['inv_type'].value_counts()
    axes[1, 0].bar(range(len(inv_type_counts)), inv_type_counts.values)
    axes[1, 0].set_title('Investment Type Distribution')
    axes[1, 0].set_xlabel('Investment Type')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_xticks(range(len(inv_type_counts)))
    axes[1, 0].set_xticklabels(inv_type_counts.index, rotation=45)

    # Goal distribution
    goal_counts = df['goal'].value_counts()
    axes[1, 1].bar(range(len(goal_counts)), goal_counts.values)
    axes[1, 1].set_title('Financial Goal Distribution')
    axes[1, 1].set_xlabel('Financial Goal')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_xticks(range(len(goal_counts)))
    axes[1, 1].set_xticklabels(goal_counts.index, rotation=45)

    # Risk score vs ROI
    scatter = axes[1, 2].scatter(df['risk_score'], df['roi'], c=true_labels, alpha=0.6, cmap='viridis')
    axes[1, 2].set_title('Risk Score vs ROI by Segment')
    axes[1, 2].set_xlabel('Risk Score')
    axes[1, 2].set_ylabel('ROI (%)')
    plt.colorbar(scatter, ax=axes[1, 2], label='Segment')

    plt.tight_layout()
    plt.savefig('outputs/dataset_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Dataset analysis plot saved to 'outputs/dataset_analysis.png'")
    plt.close()

    # Compute silhouette score for true labels
    numerical_features = ['age', 'saving', 'net_worth', 'tenure', 'duration', 'freq', 'risk_score', 'roi']
    categorical_features = ['goal', 'inv_type']
    df_encoded = df.copy()
    le_goal = LabelEncoder()
    le_inv_type = LabelEncoder()
    df_encoded['goal_encoded'] = le_goal.fit_transform(df['goal'])
    df_encoded['inv_type_encoded'] = le_inv_type.fit_transform(df['inv_type'])
    features = numerical_features + ['goal_encoded', 'inv_type_encoded']
    X = df_encoded[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    silhouette = silhouette_score(X_scaled, true_labels)
    print(f"📊 Silhouette Score for True Labels: {silhouette:.3f}")

def main():
    print("Generating optimized investment customer dataset...")
    data, true_labels = generate_customer_segments()
    data = clean_and_validate_data(data)
    df = pd.DataFrame(data)

    print("Calculating risk scores and ROI...")
    df['risk_score'] = calculate_risk_score(df['age'], df['saving'], df['freq'], df['duration'])
    df['roi'] = calculate_roi(df['risk_score'], df['freq'], df['inv_type'])
    df['true_segment'] = true_labels

    # Shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save dataset
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/investment_customers.csv', index=False)

    # Save segment mapping
    segment_mapping = {
        0: "Conservative Low-Risk",
        1: "Moderate Risk Mid-Wealth",
        2: "High-Risk Tech-Savvy",
        3: "Balanced Long-term"
    }
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/segment_mapping.json', 'w') as f:
        json.dump(segment_mapping, f, indent=4)

    create_visualization(df, true_labels)

    print("\n✅ Dataset generated successfully!")
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📁 Saved to: 'data/investment_customers.csv'")
    print(f"🎯 Number of segments: 4")

    segment_names = ["Conservative Low-Risk", "Moderate Risk Mid-Wealth", "High-Risk Tech-Savvy", "Balanced Long-term"]
    for i in range(4):
        count = sum(true_labels == i)
        percent = 100 * count / len(true_labels)
        print(f"  Segment {i}: {count} customers ({percent:.1f}%) - {segment_names[i]}")

    print(f"\n📊 Net worth range: ₹{df['net_worth'].min()/100000:.2f}L – ₹{df['net_worth'].max()/100000:.2f}L")
    print(f"💰 Savings range: ₹{df['saving'].min():,.0f} – ₹{df['saving'].max():,.0f}")
    print(f"🎯 Investment Types: {df['inv_type'].nunique()}, Goals: {df['goal'].nunique()}")
    print(f"📈 Risk Score range: {df['risk_score'].min()} – {df['risk_score'].max()}")
    print(f"📉 ROI range: {df['roi'].min()}% – {df['roi'].max()}%")

    print("\n🔧 Next Steps:")
    print("  1. Train clustering models using train_models.py")
    print("  2. Use `investment_customers.csv` in your Streamlit app")
    print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return df

if __name__ == "__main__":
    df = main()