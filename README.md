# AI-vs-Human-Text-Detection-System
AI vs Human Text Detection System 
Overview 
This project implements a supervised machine learning system to classify whether a given text is AI-generated or 
human-written. The system combines textual representations with linguistic features to improve classification 
performance and interpretability. 
Objective 
• Classify input text into AI-generated or human-written categories  
• Capture writing style differences using hybrid feature engineering  
• Provide confidence-based decision outputs for better usability  
Methodology 
Data Preprocessing 
Text data is normalized by converting to lowercase and removing punctuation. Labels are encoded into numerical 
form where AI is represented as 1 and Human as 0. 
Feature Engineering 
Textual features are extracted using TF-IDF vectorization with unigram and bigram representations to capture 
contextual patterns. In addition, linguistic features such as character count, word count, average word length, and 
punctuation density are computed to capture writing style characteristics. 
The TF-IDF feature matrix is combined with scaled numerical features using feature stacking to form a unified feature 
space. 
Data Balancing 
To address class imbalance, SMOTE (Synthetic Minority Oversampling Technique) is applied only on the training data 
to avoid data leakage. 
Models Used 
• Logistic Regression  
o Used as a baseline model  
o Effective for high-dimensional sparse data (TF-IDF)  
o Fast and interpretable  
o Provides a reference for comparison  
• Random Forest Classifier  
o Ensemble model based on multiple decision trees  
o Captures non-linear relationships in data  
o Reduces overfitting and improves generalization  
• Gradient Boosting Classifier  
o Sequential ensemble model that corrects previous errors  
o Learns complex and subtle patterns in text data  
o Improves prediction accuracy iteratively  
• XGBoost Classifier  
o Optimized implementation of gradient boosting  
o Incorporates regularization for better generalization  
o Efficient and scalable  
o Achieved the best overall performance  
Model Evaluation 
The models are evaluated using the following metrics: 
• Accuracy  
• Precision  
• Recall  
• F1-score  
Random Forest achieved the highest overall performance with balanced precision, recall, and F1-score across both 
classes, indicating strong generalization capability. 
Intelligent Decision Layer 
A rule-based decision layer is introduced to improve interpretability. Instead of directly using raw probabilities, 
predictions are categorized based on confidence levels: 
• Confidence ≥ 0.80 → Acceptable  
• 0.60 ≤ Confidence < 0.80 → Needs Review  
• Confidence < 0.60 → Uncertain  
This layer converts model outputs into meaningful and actionable insights. 
Frontend 
A Streamlit-based web interface is developed to enable real-time interaction. The application allows users to: 
• Input custom text  
• Select different machine learning models  
• Adjust confidence threshold  
• View predictions along with confidence scores  
Conclusion 
The project demonstrates that combining multiple machine learning models with hybrid feature engineering and a 
confidence-based decision layer results in a robust and interpretable system for AI-generated text detection. Random 
Forest performs best overall, while other models contribute to comparative analysis and robustness. 
Future Work 
• Hyperparameter tuning for improved performance  
• Integration with advanced NLP models  
• Deployment as a scalable web application or API 
