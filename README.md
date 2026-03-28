<p><b>AI vs Human Text Detection System</b></p>

<p><b>Overview</b></p>

This project implements a supervised machine learning system to classify whether a given text is AI-generated or human-written.  
It combines textual representations with linguistic features to improve classification performance and interpretability.

<p><b>Dataset</b></p>

The dataset used for this project was obtained from Kaggle:  
[AI vs Human Text Classification Dataset](https://www.kaggle.com/datasets/gulfan/ai-vs-human-text-classification-dataset?utm_source=chatgpt.com)

<p><b>Objectives</b></p>

- Classify input text into AI-generated or human-written categories  
- Capture writing style differences using hybrid feature engineering  
- Provide confidence-based decision outputs for better usability  

<p><b>Data Preprocessing</b></p>

- Convert text to lowercase  
- Remove punctuation  
- Encode labels: AI - 1, Human - 0  

<p><b>Feature Engineering</b></p>

- TF-IDF vectorization with unigram and bigram representations  
- Linguistic features: character count, word count, average word length, punctuation density  
- Feature stacking: combine TF-IDF matrix with scaled numerical features  

<p><b>Data Balancing</b></p>

- Applied SMOTE (Synthetic Minority Oversampling Technique) on training data to handle class imbalance  

<p><b>Models Used</b></p>

- Logistic Regression: baseline, interpretable, effective for sparse data  
- Random Forest: ensemble of decision trees, captures non-linear relationships, strong generalization  
- Gradient Boosting: sequential learning, corrects errors iteratively, improves accuracy  
- XGBoost: optimized gradient boosting, regularization, efficient and scalable  

<p><b>Model Evaluation</b></p>

Metrics used:

- Accuracy  
- Precision  
- Recall  
- F1-score  

Result: XGBoost achieved the highest overall performance with balanced precision, recall, and F1-score.

<p><b>Intelligent Decision Layer</b></p>

Predictions are categorized based on confidence levels:

- Confidence ≥ 0.80 - Acceptable  
- 0.60 ≤ Confidence < 0.80 - Needs Review  
- Confidence < 0.60 → Uncertain  

This improves interpretability and provides actionable insights.

<p><b>Frontend</b></p>

A Streamlit-based web interface allows users to:

- Input custom text  
- Select different ML models  
- Adjust confidence threshold  
- View predictions with confidence scores  

<p><b>Conclusion</b></p>

Combining multiple ML models with hybrid feature engineering and a confidence-based decision layer results in a robust and interpretable system for AI-generated text detection.  
Random Forest performs best overall, while other models contribute to comparative analysis and robustness.

<p><b>Future Work</b></p>

- Hyperparameter tuning for improved performance  
- Integration with advanced NLP models such as BERT  
- Deployment as a scalable web application or API
