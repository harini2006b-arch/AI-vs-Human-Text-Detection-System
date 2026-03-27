# AI vs Human Text Detection System
    
## Overview
This project implements a supervised machine learning system to classify whether a given text is AI-generated or human-written.  
It combines textual representations with linguistic features to improve classification performance and interpretability.
## Objectives
- Classify input text into AI-generated or human-written categories  
- Capture writing style differences using hybrid feature engineering  
- Provide confidence-based decision outputs for better usability  
### Data Preprocessing
- Convert text to lowercase  
- Remove punctuation  
- Encode labels: AI - 1, Human - 0  
### Feature Engineering
- TF-IDF vectorization with unigram and bigram representations  
- Linguistic features: character count, word count, average word length, punctuation density  
- Feature stacking: combine TF-IDF matrix with scaled numerical features  

### Data Balancing
- Applied SMOTE (Synthetic Minority Oversampling Technique) on training data to handle class imbalance  

## Models Used
- Logistic Regression: baseline, interpretable, effective for sparse data  
- Random Forest: ensemble of decision trees, captures non-linear relationships, strong generalization  
- Gradient Boosting: sequential learning, corrects errors iteratively, improves accuracy  
- XGBoost: optimized gradient boosting, regularization, efficient and scalable  

## Model Evaluation
Metrics used:
- Accuracy  
- Precision  
- Recall  
- F1-score  

Result: XGBoost achieved the highest overall performance with balanced precision, recall, and F1-score.

## Intelligent Decision Layer
Predictions are categorized based on confidence levels:
- Confidence ≥ 0.80 → Acceptable  
- 0.60 ≤ Confidence < 0.80 → Needs Review  
- Confidence < 0.60 → Uncertain  

This improves interpretability and provides actionable insights.

## Frontend
A Streamlit-based web interface allows users to:
- Input custom text  
- Select different ML models  
- Adjust confidence threshold  
- View predictions with confidence scores  

## Conclusion
Combining multiple ML models with hybrid feature engineering and a confidence-based decision layer results in a robust and interpretable system for AI-generated text detection.  
Random Forest performs best overall, while other models contribute to comparative analysis and robustness.

## Future Work
- Hyperparameter tuning for improved performance  
- Integration with advanced NLP models (e.g., BERT, RoBERTa)  
- Deployment as a scalable web application or API  



