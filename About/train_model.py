import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from detector import PhishingDetector

def create_sample_data():
    # Sample phishing emails
    phishing_emails = [
        "Urgent: Your account has been compromised! Click here to verify: http://fake-bank.xyz/login",
        "Congratulations! You've won $1,000,000. Claim now: http://prize-winner.top/claim",
        "Action required: Update your payment information immediately at http://secure-pay.bid/update",
        "Your package is delayed. Track it here: http://shipping-update.win/track",
        "Security alert: Suspicious login detected. Confirm identity: http://account-verify.xyz/confirm"
    ]

    # Sample legitimate emails
    legitimate_emails = [
        "Your order has been shipped. Track it using this link: https://amazon.com/track",
        "Meeting reminder: Team standup at 10 AM tomorrow.",
        "Invoice for your recent purchase. Payment due in 30 days.",
        "Welcome to our newsletter! Here's what's new this month.",
        "Password reset confirmation. If you didn't request this, please contact support."
    ]

    emails = phishing_emails + legitimate_emails
    labels = [1] * len(phishing_emails) + [0] * len(legitimate_emails)  # 1 for phishing, 0 for legitimate

    return emails, labels

def train_model():
    # Create sample data
    emails, labels = create_sample_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.2, random_state=42)

    # Initialize detector for preprocessing
    detector = PhishingDetector.__new__(PhishingDetector)  # Create instance without calling __init__
    detector.stop_words = set()  # Will be set properly later

    # Preprocess training data
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import nltk
    import re

    nltk.download('punkt')
    nltk.download('stopwords')
    detector.stop_words = set(stopwords.words('english'))

    X_train_processed = [detector.preprocess_text(email) for email in X_train]
    X_test_processed = [detector.preprocess_text(email) for email in X_test]

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vectorized = vectorizer.fit_transform(X_train_processed)
    X_test_vectorized = vectorizer.transform(X_test_processed)

    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_vectorized, y_train)

    # Evaluate
    y_pred = model.predict(X_test_vectorized)
    print("Model Evaluation:")
    print(classification_report(y_test, y_pred))

    # Save model and vectorizer
    import os
    os.makedirs('models', exist_ok=True)

    with open('models/phishing_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    print("Model and vectorizer saved successfully!")

if __name__ == "__main__":
    train_model()
