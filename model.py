import numpy as np
import pandas as pd
import re
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
#loading the dataset
dataset = pd.read_csv("ai_vs_human_text.csv")
dataset = dataset.drop(columns=["id"])
dataset['target'] = dataset['label'].map({'ai': 1, 'human': 0})
#preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text
dataset['text'] = dataset['text'].apply(clean_text)
#scaling numerical features
scaler = StandardScaler()
num_cols = ['char_count', 'word_count', 'avg_word_length', 'punctuation_density']
dataset[num_cols] = scaler.fit_transform(dataset[num_cols])
#for text conversion
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')
text_features = vectorizer.fit_transform(dataset['text'])
#combined preprocessed numerical and text feature
numeric_features = dataset[num_cols].values
X = hstack([text_features, numeric_features])
y = dataset['target']
#training and splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#smote
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
#model training
log_model = LogisticRegression(max_iter=1000, solver='liblinear')
rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
gb_model = GradientBoostingClassifier(random_state=42)
xgb_model = XGBClassifier(n_estimators=200,learning_rate=0.1,max_depth=6,random_state=42,eval_metric='logloss')
log_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
#create feature
def create_features(text):
    text = clean_text(text)
    text_vec = vectorizer.transform([text])
    char_len = len(text)
    words = text.split()
    word_len = len(words)
    avg_len = np.mean([len(w) for w in words]) if word_len > 0 else 0
    punct_density = sum(1 for c in text if c in ".,!?;:") / len(text) if len(text) > 0 else 0
    num_vec = pd.DataFrame([[char_len, word_len, avg_len, punct_density]],columns=num_cols)
    num_vec = scaler.transform(num_vec)
    return hstack([text_vec, num_vec])
#decision layer
def decision_logic(conf):
    if conf >= 0.80:
        return "Acceptable"
    elif conf >= 0.60:
        return "Needs Review"
    else:
        return "Uncertain / Likely AI"
#prediction
def predict(text, model_name, threshold=0.6):
    model_map = {"Logistic Regression": log_model,"Random Forest": rf_model,"Gradient Boosting": gb_model,"XGBoost": xgb_model}
    model = model_map[model_name]
    features = create_features(text)
    probs = model.predict_proba(features)[0]
    ai_index = list(model.classes_).index(1)
    confidence = probs[ai_index]
    label = "AI" if confidence >= threshold else "Human"
    decision = decision_logic(confidence)
    return label, confidence, decision

