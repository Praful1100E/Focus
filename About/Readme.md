# AI-Powered Phishing Email Detector

## Overview
This project implements an AI-powered phishing email detector using Python, Natural Language Processing (NLP), and Machine Learning (ML). Unlike traditional rule-based systems, this detector uses machine learning to analyze email content and provide a risk score for potential phishing attempts.

## Features
- **NLP Analysis**: Processes email text using tokenization, stop-word removal, and TF-IDF vectorization
- **Phishing Pattern Detection**: Identifies suspicious patterns like urgent language, fake links, and suspicious domains
- **Risk Scoring**: Provides a risk score from 0-100 indicating the likelihood of the email being phishing
- **Machine Learning Model**: Uses Logistic Regression trained on email datasets
- **Confidence Levels**: Categorizes results as Low, Medium, or High confidence
- **Batch Processing**: Analyze multiple emails at once for efficient processing of large volumes
- **Web Interface**: User-friendly web interface for easy email analysis
- **REST API**: Programmatic access via REST API endpoints for integration

## Tech Stack
- Python 3.x
- NLTK (Natural Language Toolkit)
- scikit-learn (Machine Learning)
- pandas (Data manipulation)
- NumPy (Numerical computing)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ai-phishing-detector
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model (optional, pre-trained model included):
   ```bash
   python train_model.py
   ```

## Usage

### Training the Model
Run the training script to create and save the ML model:
```bash
python train_model.py
```

### Detecting Phishing Emails
Use the detector in your Python code:
```python
from detector import PhishingDetector

detector = PhishingDetector()
result = detector.analyze_email("Your email content here")

print(f"Is Phishing: {result['is_phishing']}")
print(f"Risk Score: {result['risk_score']}/100")
print(f"Confidence: {result['confidence']}")
print(f"Patterns: {result['patterns_detected']}")
```

### Command Line Usage
Run the detector directly:
```bash
python detector.py
```

## Project Structure
```
About/
├── detector.py          # Main phishing detection class
├── train_model.py       # Model training script
├── requirements.txt     # Python dependencies
├── Readme.md           # Project documentation
├── models/             # Trained models (created after training)
│   ├── phishing_model.pkl
│   └── vectorizer.pkl
└── data/               # Sample data (if needed)
```

## How It Works
1. **Text Preprocessing**: Emails are cleaned, tokenized, and stop words are removed
2. **Feature Extraction**: TF-IDF vectorization converts text to numerical features
3. **ML Prediction**: Logistic Regression model predicts phishing probability
4. **Pattern Analysis**: Additional rule-based checks for common phishing indicators
5. **Risk Scoring**: Combines ML prediction with pattern detection for final score

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License.
