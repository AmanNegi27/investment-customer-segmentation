import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Number of customers
n_customers = 2000  # Increased for better clustering

def generate_customer_segments():
    """
    Generate 4 distinct customer segments with different characteristics
    """
    
    # Segment 0: Conservative Low-Risk Investors (25%)
    n_conservative = int(0.25 * n_customers)
    conservative_data = {
        'age': np.random.normal(50, 8, n_conservative).astype(int),  # Older customers
        'saving': np.random.normal(15000, 5000, n_conservative),     # Moderate savings
        'net_worth': np.random.normal(800000, 300000, n_conservative), # High net worth
        'tenure': np.random.normal(8, 3, n_conservative),            # Experienced
        'duration': np.random.normal(60, 20, n_conservative),        # Long-term
        'freq': np.random.normal(4, 2, n_conservative),              # Low frequency
        'goal': np.random.choice(['retirement', 'wealth'], n_conservative, p=[0.7, 0.3]),
        'inv_type': np.random.choice(['fd', 'ppf', 'bonds'], n_conservative, p=[0.4, 0.4, 0.2])
    }
    
    # Segment 1: Moderate Risk Mid-Wealth (30%)
    n_moderate = int(0.30 * n_customers)
    moderate_data = {
        'age': np.random.normal(35, 7, n_moderate).astype(int),      # Middle-aged
        'saving': np.random.normal(25000, 8000, n_moderate),         # Good savings
        'net_worth': np.random.normal(400000, 200000, n_moderate),   # Moderate net worth
        'tenure': np.random.normal(5, 2, n_moderate),                # Some experience
        'duration': np.random.normal(36, 15, n_moderate),            # Medium-term
        'freq': np.random.normal(8, 3, n_moderate),                  # Moderate frequency
        'goal': np.random.choice(['wealth', 'education', 'retirement'], n_moderate, p=[0.4, 0.4, 0.2]),
        'inv_type': np.random.choice(['mutual', 'etf', 'gold'], n_moderate, p=[0.5, 0.3, 0.2])
    }
    
    # Segment 2: High-Risk Tech-Savvy (25%)
    n_aggressive = int(0.25 * n_customers)
    aggressive_data = {
        'age': np.random.normal(28, 5, n_aggressive).astype(int),    # Young investors
        'saving': np.random.normal(35000, 12000, n_aggressive),      # High savings
        'net_worth': np.random.normal(200000, 150000, n_aggressive), # Lower but growing
        'tenure': np.random.normal(3, 1.5, n_aggressive),            # New to investing
        'duration': np.random.normal(18, 8, n_aggressive),           # Short-term
        'freq': np.random.normal(15, 5, n_aggressive),               # High frequency
        'goal': np.random.choice(['wealth', 'vacation', 'education'], n_aggressive, p=[0.6, 0.2, 0.2]),
        'inv_type': np.random.choice(['stocks', 'crypto', 'etf'], n_aggressive, p=[0.4, 0.4, 0.2])
    }
    
    # Segment 3: Balanced Long-term Wealth Builders (20%)
    n_balanced = n_customers - n_conservative - n_moderate - n_aggressive
    balanced_data = {
        'age': np.random.normal(42, 6, n_balanced).astype(int),      # Mature investors
        'saving': np.random.normal(45000, 15000, n_balanced),        # High savings
        'net_worth': np.random.normal(1200000, 500000, n_balanced),  # High net worth
        'tenure': np.random.normal(12, 4, n_balanced),               # Very experienced
        'duration': np.random.normal(84, 30, n_balanced),            # Very long-term
        'freq': np.random.normal(6, 2, n_balanced),                  # Steady frequency
        'goal': np.random.choice(['retirement', 'wealth', 'emergency'], n_balanced, p=[0.5, 0.4, 0.1]),
        'inv_type': np.random.choice(['real_estate', 'mutual', 'stocks', 'nps'], n_balanced, p=[0.3, 0.3, 0.2, 0.2])
    }
    
    # Combine all segments
    all_data = {}
    for key in conservative_data.keys():
        all_data[key] = np.concatenate([
            conservative_data[key],
            moderate_data[key],
            aggressive_data[key],
            balanced_data[key]
        ])
    
    # Add true cluster labels for validation
    true_labels = np.concatenate([
        np.zeros(n_conservative),
        np.ones(n_moderate),
        np.full(n_aggressive, 2),
        np.full(n_balanced, 3)
    ])
    
    return all_data, true_labels

def clean_and_validate_data(data):
    """Clean and validate the generated data"""
    
    # Ensure age is within reasonable bounds
    data['age'] = np.clip(data['age'], 18, 75)
    
    # Ensure savings are positive
    data['saving'] = np.clip(data['saving'], 5000, 100000)
    
    # Ensure net worth is positive and reasonable
    data['net_worth'] = np.clip(data['net_worth'], 50000, 5000000)
    
    # Ensure tenure is non-negative
    data['tenure'] = np.clip(data['tenure'], 0, 25).astype(int)
    
    # Ensure duration is positive
    data['duration'] = np.clip(data['duration'], 3, 120).astype(int)
    
    # Ensure frequency is positive
    data['freq'] = np.clip(data['freq'], 1, 24).astype(int)
    
    return data

def calculate_risk_score(age, saving, freq, duration):
    """Calculate risk score using the same formula as your app"""
    score = (age * 0.3) + (saving / 8000) + (freq * 1.2) - (duration * 0.1)
    return np.clip(score, 0, 100).round(1)

def calculate_roi(risk_score, freq, inv_type):
    """Calculate ROI using the same formula as your app"""
    base_roi = 5
    risk_bonus = risk_score / 20
    freq_bonus = freq * 0.1
    
    inv_type_bonus_map = {
        'stocks': 5, 'mutual': 4, 'crypto': 8, 'gold': 2,
        'fd': 1.5, 'ppf': 2, 'bonds': 2.5, 'real_estate': 3,
        'etf': 4, 'nps': 2
    }
    
    # Vectorized calculation
    inv_type_bonus = np.array([inv_type_bonus_map.get(inv, 2) for inv in inv_type])
    roi = base_roi + risk_bonus + freq_bonus + inv_type_bonus
    
    return roi.round(2)

def create_visualization(df, true_labels):
    """Create visualizations to show data distribution"""
    
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Investment Customer Dataset - Segment Analysis', fontsize=16)
    
    # Age distribution by segment
    for i in range(4):
        segment_data = df[true_labels == i]
        axes[0, 0].hist(segment_data['age'], alpha=0.7, label=f'Segment {i}', bins=20)
    axes[0, 0].set_title('Age Distribution by Segment')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # Risk Score distribution by segment
    for i in range(4):
        segment_data = df[true_labels == i]
        axes[0, 1].hist(segment_data['risk_score'], alpha=0.7, label=f'Segment {i}', bins=20)
    axes[0, 1].set_title('Risk Score Distribution by Segment')
    axes[0, 1].set_xlabel('Risk Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # Net Worth distribution by segment
    for i in range(4):
        segment_data = df[true_labels == i]
        axes[0, 2].hist(segment_data['net_worth']/1000, alpha=0.7, label=f'Segment {i}', bins=20)
    axes[0, 2].set_title('Net Worth Distribution by Segment (in thousands)')
    axes[0, 2].set_xlabel('Net Worth (₹ thousands)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()
    
    # Investment Type distribution
    inv_type_counts = df.groupby(['inv_type']).size().sort_values(ascending=False)
    axes[1, 0].bar(range(len(inv_type_counts)), inv_type_counts.values)
    axes[1, 0].set_title('Investment Type Distribution')
    axes[1, 0].set_xlabel('Investment Type')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_xticks(range(len(inv_type_counts)))
    axes[1, 0].set_xticklabels(inv_type_counts.index, rotation=45)
    
    # Goal distribution
    goal_counts = df.groupby(['goal']).size().sort_values(ascending=False)
    axes[1, 1].bar(range(len(goal_counts)), goal_counts.values)
    axes[1, 1].set_title('Financial Goal Distribution')
    axes[1, 1].set_xlabel('Financial Goal')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_xticks(range(len(goal_counts)))
    axes[1, 1].set_xticklabels(goal_counts.index, rotation=45)
    
    # Risk Score vs ROI scatter
    scatter = axes[1, 2].scatter(df['risk_score'], df['roi'], c=true_labels, alpha=0.6, cmap='viridis')
    axes[1, 2].set_title('Risk Score vs ROI by Segment')
    axes[1, 2].set_xlabel('Risk Score')
    axes[1, 2].set_ylabel('ROI (%)')
    plt.colorbar(scatter, ax=axes[1, 2], label='Segment')
    
    plt.tight_layout()
    plt.savefig('outputs/dataset_analysis.png', dpi=300, bbox_inches='tight')
    print("Dataset analysis plot saved to 'outputs/dataset_analysis.png'")
    plt.close()

def main():
    print("Generating improved investment customer dataset...")
    
    # Generate segmented data
    data, true_labels = generate_customer_segments()
    
    # Clean and validate data
    data = clean_and_validate_data(data)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Calculate risk scores and ROI
    print("Calculating risk scores and ROI...")
    df['risk_score'] = calculate_risk_score(df['age'], df['saving'], df['freq'], df['duration'])
    df['roi'] = calculate_roi(df['risk_score'], df['freq'], df['inv_type'])
    
    # Add true segment labels for reference
    df['true_segment'] = true_labels.astype(int)
    
    # Shuffle the dataset to mix segments
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Create output directory
    os.makedirs('data', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # Save the dataset
    df.to_csv('data/investment_customers.csv', index=False)
    
    # Create visualization
    create_visualization(df, true_labels)
    
    # Print dataset summary
    print(f"\n✅ Dataset generated successfully!")
    print(f"📊 Dataset shape: {df.shape}")
    print(f"🎯 Number of segments: 4")
    print(f"📁 Saved to: 'data/investment_customers.csv'")
    
    print(f"\n📈 Segment Distribution:")
    segment_counts = pd.Series(true_labels).value_counts().sort_index()
    segment_names = [
        "Conservative Low-Risk",
        "Moderate Risk Mid-Wealth", 
        "High-Risk Tech-Savvy",
        "Balanced Long-term"
    ]
    
    for i, (count, name) in enumerate(zip(segment_counts, segment_names)):
        percentage = (count / len(true_labels)) * 100
        print(f"  Segment {i}: {count} customers ({percentage:.1f}%) - {name}")
    
    print(f"\n📊 Key Statistics:")
    print(f"  Age range: {df['age'].min()} - {df['age'].max()}")
    print(f"  Savings range: ₹{df['saving'].min():,.0f} - ₹{df['saving'].max():,.0f}")
    print(f"  Net worth range: ₹{df['net_worth'].min():,.0f} - ₹{df['net_worth'].max():,.0f}")
    print(f"  Risk score range: {df['risk_score'].min()} - {df['risk_score'].max()}")
    print(f"  ROI range: {df['roi'].min()}% - {df['roi'].max()}%")
    
    print(f"\n🎯 Investment Types: {df['inv_type'].nunique()} unique types")
    print(f"🎯 Financial Goals: {df['goal'].nunique()} unique goals")
    
    print(f"\n🔧 Next Steps:")
    print(f"  1. Run this script to generate your dataset")
    print(f"  2. Use the generated 'data/investment_customers.csv' for training")
    print(f"  3. Train your clustering models (KMeans, GMM)")
    print(f"  4. Run your Streamlit app for customer segmentation")
    
    return df

if __name__ == "__main__":
    df = main()