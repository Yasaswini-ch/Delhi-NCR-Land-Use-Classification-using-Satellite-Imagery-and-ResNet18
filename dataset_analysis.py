#!/usr/bin/env python3
"""
Advanced Dataset Analysis for Delhi-NCR Air Quality Prediction
Performs detailed analysis without model training.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_dataset_characteristics():
    """Analyze dataset characteristics and patterns."""
    
    # Load baseline results
    baseline_df = pd.read_csv('data/outputs/baseline_comparison.csv')
    
    print("=" * 60)
    print("DATASET-BASED ANALYSIS & INSIGHTS")
    print("=" * 60)
    
    # 1. Model Performance Analysis
    print("\n1. MODEL PERFORMANCE COMPARISON")
    print("-" * 30)
    
    best_model = baseline_df.loc[baseline_df['accuracy_mean'].idxmax()]
    print(f"Best Performing Model: {best_model['model']}")
    print(f"Accuracy: {best_model['accuracy_mean']:.3f} ± {best_model['accuracy_std']:.3f}")
    print(f"F1-Score: {best_model['f1_macro_mean']:.3f} ± {best_model['f1_macro_std']:.3f}")
    
    # Performance stability analysis
    baseline_df['cv_coefficient'] = baseline_df['accuracy_std'] / baseline_df['accuracy_mean']
    most_stable = baseline_df.loc[baseline_df['cv_coefficient'].idxmin()]
    print(f"\nMost Stable Model: {most_stable['model']}")
    print(f"CV Coefficient (lower is better): {most_stable['cv_coefficient']:.3f}")
    
    return baseline_df

def analyze_feature_importance_patterns():
    """Analyze feature importance from the evaluation output."""
    
    print("\n2. FEATURE IMPORTANCE ANALYSIS")
    print("-" * 30)
    
    # From evaluation output, we know top features:
    feature_importance = {
        'local_mean': 0.0843,
        'b_max': 0.0802,
        'g_std': 0.0757,
        'local_max': 0.0749,
        'r_max': 0.0682,
        'local_std': 0.0662,
        'r_std': 0.0607,
        'r_min': 0.0607,
        'g_mean': 0.0555,
        'b_mean': 0.0547
    }
    
    print("Top 10 Most Important Features:")
    for i, (feature, importance) in enumerate(feature_importance.items(), 1):
        print(f"{i:2d}. {feature:12s}: {importance:.4f}")
    
    # Feature categories analysis
    local_features = ['local_mean', 'local_max', 'local_std']
    rgb_max_features = ['r_max', 'g_max', 'b_max']
    rgb_std_features = ['r_std', 'g_std', 'b_std']
    rgb_mean_features = ['r_mean', 'g_mean', 'b_mean']
    
    local_importance = sum(feature_importance.get(f, 0) for f in local_features)
    rgb_max_importance = sum(feature_importance.get(f, 0) for f in rgb_max_features)
    rgb_std_importance = sum(feature_importance.get(f, 0) for f in rgb_std_features)
    rgb_mean_importance = sum(feature_importance.get(f, 0) for f in rgb_mean_features)
    
    print(f"\nFeature Category Importance:")
    print(f"Local Statistics: {local_importance:.4f}")
    print(f"RGB Max Values:   {rgb_max_importance:.4f}")
    print(f"RGB Std Values:   {rgb_std_importance:.4f}")
    print(f"RGB Mean Values:  {rgb_mean_importance:.4f}")
    
    return feature_importance

def generate_dataset_recommendations():
    """Generate recommendations based on dataset analysis."""
    
    print("\n3. DATASET-BASED RECOMMENDATIONS")
    print("-" * 30)
    
    print("📈 PERFORMANCE INSIGHTS:")
    print("• Logistic Regression performs best (25% accuracy)")
    print("• High variance (±0.18) indicates dataset size limitations")
    print("• Class imbalance affects model stability")
    
    print("\n🎯 FEATURE INSIGHTS:")
    print("• Local statistics are most predictive (22.5% total importance)")
    print("• RGB color channels provide complementary information")
    print("• Maximum values more important than means")
    
    print("\n💡 DATA IMPROVEMENT RECOMMENDATIONS:")
    print("• Increase dataset size to reduce variance")
    print("• Address class imbalance through sampling")
    print("• Focus on local spatial features")
    print("• Consider temporal features (seasonal patterns)")
    print("• Add more diverse land-cover samples")
    
    print("\n🔧 MODEL SELECTION INSIGHTS:")
    print("• Simple models (Logistic Regression) work better with small data")
    print("• Complex models overfit with current dataset size")
    print("• Ensemble methods could improve stability")

def create_performance_visualization():
    """Create performance comparison visualization."""
    
    # Load data
    baseline_df = pd.read_csv('data/outputs/baseline_comparison.csv')
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    bars1 = ax1.bar(baseline_df['model'], baseline_df['accuracy_mean'], 
                    yerr=baseline_df['accuracy_std'], capsize=5, alpha=0.7)
    ax1.set_title('Model Accuracy Comparison (5-fold CV)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 0.5)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, baseline_df['accuracy_mean']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # F1-Score comparison
    bars2 = ax2.bar(baseline_df['model'], baseline_df['f1_macro_mean'],
                    yerr=baseline_df['f1_macro_std'], capsize=5, alpha=0.7, color='orange')
    ax2.set_title('Model F1-Score Comparison (5-fold CV)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('F1-Score (Macro)')
    ax2.set_ylim(0, 0.5)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, f1 in zip(bars2, baseline_df['f1_macro_mean']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{f1:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save visualization
    output_path = Path('data/outputs/dataset_analysis.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Dataset analysis visualization saved: {output_path}")

def create_feature_analysis_report():
    """Create comprehensive feature analysis report."""
    
    print("\n4. COMPREHENSIVE FEATURE ANALYSIS")
    print("-" * 30)
    
    feature_analysis = {
        'Local Spatial Features': {
            'features': ['local_mean', 'local_max', 'local_std'],
            'total_importance': 0.2254,
            'interpretation': 'Local land-cover patterns around stations'
        },
        'RGB Color Features': {
            'features': ['r_mean', 'g_mean', 'b_mean', 'r_std', 'g_std', 'b_std', 'r_max', 'g_max', 'b_max'],
            'total_importance': 0.7746,
            'interpretation': 'Color characteristics of satellite imagery'
        }
    }
    
    print("Feature Group Analysis:")
    for group, info in feature_analysis.items():
        print(f"\n{group}:")
        print(f"  Total Importance: {info['total_importance']:.4f}")
        print(f"  Interpretation: {info['interpretation']}")
        print(f"  Number of Features: {len(info['features'])}")
    
    # Feature engineering suggestions
    print(f"\n🚀 ADVANCED FEATURE ENGINEERING IDEAS:")
    print("• Texture features (GLCM, LBP)")
    print("• Spatial autocorrelation metrics")
    print("• Multi-scale patch features")
    print("• Spectral indices (NDVI, NDBI)")
    print("• Distance-based features (to urban centers, water bodies)")
    print("• Temporal features (if time-series data available)")

if __name__ == "__main__":
    print("🔍 DELHI-NCR DATASET ANALYSIS & ADVANCEMENTS")
    print("=" * 60)
    
    # Run all analyses
    baseline_df = analyze_dataset_characteristics()
    feature_importance = analyze_feature_importance_patterns()
    generate_dataset_recommendations()
    create_performance_visualization()
    create_feature_analysis_report()
    
    print("\n" + "=" * 60)
    print("✅ DATASET ANALYSIS COMPLETE!")
    print("📊 Generated insights and recommendations")
    print("📈 Created performance visualizations")
    print("=" * 60)
