import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

class PhishingDetector:
    def __init__(self, model_path='models/phishing_model.pkl', vectorizer_path='models/vectorizer.pkl'):
        # Download NLTK data if not already downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        self.stop_words = set(stopwords.words('english'))

        # Load model and vectorizer
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Model or vectorizer file not found. Please run train_model.py first.")

    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stop words
        tokens = [word for word in tokens if word not in self.stop_words]
        # Join back to string
        return ' '.join(tokens)

    def extract_features(self, email_text):
        # Preprocess the text
        processed_text = self.preprocess_text(email_text)
        # Vectorize
        features = self.vectorizer.transform([processed_text])
        return features

    def detect_phishing(self, email_text):
        # Extract features
        features = self.extract_features(email_text)
        # Predict probability
        prob = self.model.predict_proba(features)[0][1]  # Probability of being phishing
        # Predict class
        prediction = self.model.predict(features)[0]
        # Calculate risk score (0-100)
        risk_score = int(prob * 100)
        return prediction, risk_score

    def analyze_email(self, email_text):
        prediction, risk_score = self.detect_phishing(email_text)

        result = {
            'is_phishing': bool(prediction),
            'risk_score': risk_score,
            'confidence': 'High' if risk_score > 70 else 'Medium' if risk_score > 40 else 'Low'
        }

        # Additional analysis
        result['patterns_detected'] = self.detect_patterns(email_text)

        return result

    def detect_patterns(self, email_text):
        patterns = []

        # Check for urgent language
        urgent_words = ['urgent', 'immediate', 'action required', 'act now', 'limited time']
        if any(word in email_text.lower() for word in urgent_words):
            patterns.append('Urgent tone detected')

        # Check for suspicious domains (simplified)
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', email_text)
        suspicious_domains = ['.xyz', '.top', '.bid', '.win', '.club']
        for url in urls:
            if any(domain in url for domain in suspicious_domains):
                patterns.append(f'Suspicious domain in URL: {url}')

        # Check for fake links
        if 'http' in email_text and ('login' in email_text.lower() or 'verify' in email_text.lower()):
            patterns.append('Potential fake login/verification link')

        return patterns

if __name__ == "__main__":
    # Example usage
    detector = PhishingDetector()

    sample_email = """
    Urgent: Your account has been compromised! Click here to verify: http://fake-bank.xyz/login
    """

    result = detector.analyze_email(sample_email)
    print("Analysis Result:")
    print(f"Is Phishing: {result['is_phishing']}")
    print(f"Risk Score: {result['risk_score']}/100")
    print(f"Confidence: {result['confidence']}")
    print(f"Patterns Detected: {result['patterns_detected']}")
