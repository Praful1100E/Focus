import pickle
import re
import json
import csv
from datetime import datetime
from email.utils import parsedate_to_datetime
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

        # Fix vectorizer if not fitted (sklearn version compatibility issue)
        if not hasattr(self.vectorizer, 'idf_') or self.vectorizer.idf_ is None:
            # Disable IDF to avoid fitting issues
            self.vectorizer.use_idf = False
            self.vectorizer.norm = None
            if hasattr(self.vectorizer, '_tfidf'):
                self.vectorizer._tfidf.use_idf = False

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
        try:
            features = self.vectorizer.transform([processed_text])
        except Exception as e:
            if 'idf vector is not fitted' in str(e):
                # If IDF is not fitted, temporarily disable it for this transform
                original_use_idf = self.vectorizer.use_idf
                self.vectorizer.use_idf = False
                try:
                    features = self.vectorizer.transform([processed_text])
                finally:
                    self.vectorizer.use_idf = original_use_idf
            else:
                raise e
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

    def analyze_emails_batch(self, email_texts):
        """
        Analyze multiple emails at once and return results for each.

        Args:
            email_texts (list): List of email text strings to analyze

        Returns:
            list: List of analysis results, one for each email
        """
        results = []
        for email_text in email_texts:
            result = self.analyze_email(email_text)
            results.append(result)
        return results

    def export_single_result(self, result, format_type='json', filename=None):
        """
        Export a single analysis result to CSV or JSON format.

        Args:
            result (dict): Analysis result from analyze_email
            format_type (str): 'json' or 'csv'
            filename (str): Optional custom filename

        Returns:
            str: Path to the exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"phishing_analysis_single_{timestamp}"

        if format_type.lower() == 'json':
            filepath = f"{filename}.json"
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2)
        elif format_type.lower() == 'csv':
            filepath = f"{filename}.csv"
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['is_phishing', 'risk_score', 'confidence', 'patterns_detected'])
                patterns_str = '; '.join(result['patterns_detected'])
                writer.writerow([
                    result['is_phishing'],
                    result['risk_score'],
                    result['confidence'],
                    patterns_str
                ])
        else:
            raise ValueError("format_type must be 'json' or 'csv'")

        return filepath

    def export_batch_results(self, results, format_type='json', filename=None):
        """
        Export batch analysis results to CSV or JSON format.

        Args:
            results (list): List of analysis results from analyze_emails_batch
            format_type (str): 'json' or 'csv'
            filename (str): Optional custom filename

        Returns:
            str: Path to the exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"phishing_analysis_batch_{timestamp}"

        if format_type.lower() == 'json':
            filepath = f"{filename}.json"
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
        elif format_type.lower() == 'csv':
            filepath = f"{filename}.csv"
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['email_index', 'is_phishing', 'risk_score', 'confidence', 'patterns_detected'])
                for i, result in enumerate(results):
                    patterns_str = '; '.join(result['patterns_detected'])
                    writer.writerow([
                        i + 1,
                        result['is_phishing'],
                        result['risk_score'],
                        result['confidence'],
                        patterns_str
                    ])
        else:
            raise ValueError("format_type must be 'json' or 'csv'")

        return filepath

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

    def analyze_email_headers(self, headers_raw):
        """
        Analyze email headers for suspicious patterns.

        Args:
            headers_raw (str): Raw email headers as a string

        Returns:
            dict: Header analysis results
        """
        header_patterns = []
        header_info = {}

        # Parse headers into key-value pairs
        headers = {}
        for line in headers_raw.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                headers[key.strip().lower()] = value.strip()

        # Extract key header information
        header_info['from'] = headers.get('from', '')
        header_info['to'] = headers.get('to', '')
        header_info['subject'] = headers.get('subject', '')
        header_info['date'] = headers.get('date', '')
        header_info['received'] = headers.get('received', '')
        header_info['user_agent'] = headers.get('user-agent', '')
        header_info['content_type'] = headers.get('content-type', '')

        # Check for spoofed sender (From vs Return-Path mismatch)
        return_path = headers.get('return-path', '')
        if return_path and header_info['from']:
            from_domain = re.search(r'@([\w.-]+)', header_info['from'])
            return_domain = re.search(r'@([\w.-]+)', return_path)
            if from_domain and return_domain and from_domain.group(1) != return_domain.group(1):
                header_patterns.append('Sender address mismatch (possible spoofing)')

        # Check for suspicious User-Agent
        if header_info['user_agent']:
            suspicious_uas = ['phishing', 'spam', 'bot', 'crawler']
            if any(ua.lower() in header_info['user_agent'].lower() for ua in suspicious_uas):
                header_patterns.append('Suspicious User-Agent detected')

        # Check for unusual routing (too many Received headers)
        received_count = len([h for h in headers_raw.split('\n') if h.lower().startswith('received:')])
        if received_count > 5:
            header_patterns.append('Unusual email routing (too many hops)')

        # Check for forged timestamps (date too far in future/past)
        if header_info['date']:
            try:
                email_date = parsedate_to_datetime(header_info['date'])
                now = datetime.now(email_date.tzinfo)
                time_diff = abs((now - email_date).total_seconds())
                if time_diff > 86400 * 30:  # More than 30 days difference
                    header_patterns.append('Suspicious timestamp (date too far from current time)')
            except:
                header_patterns.append('Invalid or malformed date header')

        # Check for missing or suspicious Content-Type
        if not header_info['content_type'] or 'multipart/mixed' in header_info['content_type'].lower():
            if 'attachment' in headers_raw.lower():
                header_patterns.append('Potential executable attachment detected')

        return {
            'header_info': header_info,
            'header_patterns': header_patterns
        }

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
