# Financial Analytics & Fraud Detection System

A dual-analysis machine learning project combining customer segmentation and fraud detection on credit card transaction data.

## Project Overview

This project demonstrates end-to-end data analysis and machine learning capabilities through two integrated analyses:

1. **Customer Segmentation**: Identifying distinct customer groups for targeted business strategies
2. **Fraud Detection**: Building a classification model to detect fraudulent transactions in highly imbalanced data

## Technical Stack

- **Python 3.x**
- **pandas & NumPy**: Data manipulation and analysis
- **scikit-learn**: Machine learning (K-means clustering, Random Forest classification)
- **matplotlib & seaborn**: Data visualization

## Dataset

- **Customer Segmentation**: 8,950 credit card customers with 18 behavioral features
- **Fraud Detection**: 284,807 transactions with 0.17% fraud rate (492 fraudulent cases)

## Part 1: Customer Segmentation

### Approach

Applied K-means clustering to identify natural customer groupings based on:
- Credit limits and balances
- Purchase behaviors
- Payment patterns
- Cash advance usage

### Key Findings

**Identified 3 distinct customer segments:**

1. **High-Value Spenders (14.2%)**
   - High credit limits ($9,246 avg)
   - Frequent purchases ($2,009 avg)
   - Full payment behavior (81% pay in full)
   - Premium customer segment

2. **Low Activity Users (68.3%)**
   - Minimal engagement
   - Low balances and purchases
   - Retention risk group

3. **Cash Advance Users (17.4%)**
   - Heavy cash advance reliance
   - Higher credit utilization
   - Only 3% pay in full monthly
   - Higher fraud risk profile

### Business Impact

- Targeted retention strategies for each segment
- Customized product offerings based on behavior
- Risk-based fraud monitoring by segment

## Part 2: Fraud Detection

### Challenge

Extreme class imbalance: 99.83% legitimate vs 0.17% fraudulent transactions

### Approach

- Random Forest classifier with balanced class weights
- Optimized for precision-recall tradeoff rather than accuracy
- Proper train/test split validation (80/20)

### Results

**Model Performance:**
- **Precision**: 96.05% (high confidence in fraud flags)
- **Recall**: 74.49% (detected 73 out of 98 test frauds)
- **F1-Score**: 0.84
- **ROC-AUC**: 0.95 (excellent discrimination)
- **False Alarm Rate**: 0.01% (only 3 false positives out of 56,864 legitimate transactions)

### Business Impact

- Catches 3 out of 4 frauds with minimal customer friction
- Extremely low false alarm rate preserves customer experience
- Production-ready model with clear deployment strategy

## Key Technical Achievements

1. **Handled extreme class imbalance** using balanced class weights
2. **Feature standardization** for distance-based algorithms
3. **Proper model validation** through train/test methodology
4. **Business-focused metrics** (precision/recall over accuracy)
5. **Integrated analysis** connecting segmentation insights to fraud patterns

## Project Insights

- Cash Advance Users (Cluster 2) show elevated fraud risk
- Fraud transactions tend to be smaller amounts (test transactions)
- Segment-specific fraud thresholds could improve detection
- Model demonstrates real-world deployment viability

## Files

- `customer_segmentation_fraud_detection.ipynb`: Complete analysis notebook
- `elbow_method.png`: Optimal cluster determination
- `customer_segments.png`: Visual representation of customer clusters
- `fraud_patterns.png`: Fraud vs legitimate transaction patterns

## Future Enhancements

- Deep learning approaches for fraud detection
- Real-time deployment pipeline
- Segment-specific fraud models
- Feature importance analysis
- Temporal pattern analysis

## About

This project was developed to demonstrate practical data analytics and machine learning skills applicable to financial services, risk management, and business intelligence roles.

**Skills Demonstrated**: Python, pandas, scikit-learn, data cleaning, feature engineering, clustering, classification, imbalanced data handling, model evaluation, business strategy development

---

*Note: Dataset sources are publicly available from Kaggle. Data files not included in repository due to size.*
