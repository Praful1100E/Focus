from flask import Flask, render_template, request, jsonify
from detector import PhishingDetector

app = Flask(__name__)
detector = PhishingDetector()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    email_text = request.form.get('email_text')
    if not email_text:
        return render_template('index.html', error="Please enter email text.")

    result = detector.analyze_email(email_text)
    return render_template('index.html', result=result, email_text=email_text)

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    data = request.get_json()
    email_text = data.get('email_text')
    if not email_text:
        return jsonify({'error': 'email_text is required'}), 400

    result = detector.analyze_email(email_text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
