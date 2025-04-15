from flask import Flask, request, render_template, jsonify, send_file
from transformers import pipeline
from langdetect import detect
from googletrans import Translator
import re
import emoji
import pandas as pd
from collections import defaultdict
from image_detection import process_image
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import io
import uuid
import os
import json  

app = Flask(__name__)

# Load dataset
try:
    df = pd.read_csv("Final.csv", encoding="ISO-8859-1")
    # dataset.dropna(inplace=True)
    print("✔️ Dataset loaded successfully!")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    dataset = None


# Initialize Translator
translator = Translator()

# Toxic Labels and other constants
TOXIC_LABELS = [
    "TOXIC", "INSULT", "THREAT", "OBSCENE", "IDENTITY_HATE",
    "SEXUAL_EXPLOITATION", "SEVERE_TOXICITY", "PROFANITY", "HARASSMENT",
    "HATE", "BULLYING", "SARCASM"
]

BULLYING_THRESHOLD = 0.65
SENTIMENT_WEIGHTS = {"negative": 0.7, "positive": -0.3, "neutral": 0.1}
EMOTION_WEIGHTS = {"anger": 0.8, "disgust": 0.7, "fear": 0.3, "joy": -0.2, "sadness": 0.3}

# Load NLP Models
try:
    MODELS = {
        'toxicity': pipeline("text-classification",
                              model="unitary/toxic-bert", top_k=None),

        'sentiment': pipeline("sentiment-analysis", 
                              model="cardiffnlp/twitter-roberta-base-sentiment"),

        'emotion': pipeline("text-classification", 
                            model="bhadresh-savani/distilbert-base-uncased-emotion"),
                            
        'sarcasm': pipeline("text-classification", 
                            model="helinivan/english-sarcasm-detector")
    }
    
    print("✔️ All models loaded successfully!")
except Exception as e:
    print(f"❌ Model loading error: {e}")
    exit()

def preprocess_text(text):
    """Convert emojis and expand abbreviations"""
    text = emoji.demojize(text).replace(":", " ")
    abbreviations = {
        "lol": "laughing out loud",
        "smh": "shaking my head",
        "wtf": "what the fuck",
        "stfu": "shut the fuck up",
        "tf": "the fuck"
    }
    words = text.split()
    words = [abbreviations.get(word.lower(), word) for word in words]
    return " ".join(words)

from googletrans import Translator, LANGUAGES

def translate_text(text):
    """Improved language detection and translation with better error handling"""
    try:
        # First perform language detection
        translator = Translator()
        detection = translator.detect(text)
        
        # Get full language name from detected code
        lang_code = detection.lang.split('-')[0]  # Handle zh-CN, en-US etc.
        lang_name = LANGUAGES.get(lang_code, 'unknown').capitalize()
        
        # Only translate if not English and confidence is high
        if lang_code != 'en' and detection.confidence > 0.7:
            translated = translator.translate(text, dest='en').text
            return translated, lang_name
        return text, lang_name
        
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text, 'unknown'

def analyze_text(text):
    """Perform toxicity, sentiment, emotion, and sarcasm analysis"""
    results = {}
    cleaned_text = preprocess_text(text)

    # Toxicity Detection
    toxicity_results = MODELS['toxicity'](cleaned_text)[0]
    results['toxicity'] = {res['label'].upper(): res['score'] for res in toxicity_results}

    # Sentiment Analysis
    sentiment = MODELS['sentiment'](cleaned_text)[0]
    results['sentiment'] = {
        'label': sentiment['label'],
        'score': sentiment['score'] * SENTIMENT_WEIGHTS.get(sentiment['label'].lower(), 0)
    }

    # Emotion Detection
    emotions = MODELS['emotion'](cleaned_text)
    results['emotion'] = [{
        'label': e['label'],
        'score': e['score'] * EMOTION_WEIGHTS.get(e['label'].lower(), 0)
    } for e in emotions]

    # Sarcasm Detection
    sarcasm = MODELS['sarcasm'](cleaned_text)[0]
    results['sarcasm'] = {
        'label': sarcasm['label'],
        'score': sarcasm['score'] * 0.8 if sarcasm['label'] == 'SARCASTIC' else 0
    }

    return results

def detect_cyberbullying(text):
    """Main function to detect cyberbullying"""
    if not text.strip():
        return {"error": "Empty input"}

    translated_text, lang = translate_text(text)
    model_results = analyze_text(translated_text)

    final_scores = defaultdict(float)

    # Aggregate Toxicity Scores
    for label, score in model_results['toxicity'].items():
        final_scores[label] += score * 0.7

    # Adjust score using sentiment and emotions
    final_scores['BULLYING'] += model_results['sentiment']['score']
    for emotion in model_results['emotion']:
        final_scores['BULLYING'] += emotion['score']

    # Include sarcasm impact
    final_scores['SARCASM'] = model_results['sarcasm']['score']
    if model_results['sarcasm']['label'] == 'SARCASTIC':
        final_scores['BULLYING'] += model_results['sarcasm']['score']

    # Identify toxic labels above threshold
    bullying_labels = [
        (label, score)
        for label, score in final_scores.items()
        if score >= BULLYING_THRESHOLD and label in TOXIC_LABELS
    ]

    # Assign Confidence Grade
    bullying_score = final_scores.get('BULLYING', 0)
    if bullying_score >= 0.86:
        confidence = "🔴 Very High"
    elif bullying_score >= 0.71:
        confidence = "🟠 High"
    elif bullying_score >= 0.51:
        confidence = "🟡 Moderate"
    elif bullying_score >= 0.31:
        confidence = "🟢 Low"
    else:
        confidence = "✅ Very Low"

    # Generate summary
    summary = generate_summary(bullying_labels, model_results)

    return {
        "text": text,
        "language": lang,
        "scores": dict(final_scores),
        "bullying_labels": bullying_labels,
        "sentiment": model_results['sentiment'],
        "emotions": model_results['emotion'],
        "sarcasm": model_results['sarcasm'],
        "summary": summary,
        "verdict": "⚠️ Cyberbullying Detected" if bullying_labels else "✅ Clean Content",
        "confidence_grade": confidence
    }

def generate_summary(labels, results):
    """Generate a comprehensive summary report based on findings"""
    summary_parts = []
    
    # Check for specific toxic labels
    toxic_labels = [label[0] for label in labels]
    
    if 'TOXIC' in toxic_labels:
        summary_parts.append("Toxic language detected")
    if 'INSULT' in toxic_labels:
        summary_parts.append("Insulting content found")
    if 'THREAT' in toxic_labels:
        summary_parts.append("Threatening language identified")
    if 'OBSCENE' in toxic_labels:
        summary_parts.append("Obscene or vulgar content present")
    if 'IDENTITY_HATE' in toxic_labels:
        summary_parts.append("Hate speech targeting identity groups detected")
    if 'HARASSMENT' in toxic_labels:
        summary_parts.append("Content appears harassing")
    if 'SARCASM' in toxic_labels:
        summary_parts.append("Sarcastic tone amplifying negativity")
    
    # Analyze sentiment
    sentiment = results['sentiment']
    if sentiment['label'] == 'negative' and sentiment['score'] > 0.5:
        summary_parts.append("Strong negative sentiment")
    elif sentiment['label'] == 'negative':
        summary_parts.append("Negative sentiment detected")
    
    # Analyze emotions
    emotions = results['emotion']
    primary_emotion = max(emotions, key=lambda x: x['score'])
    
    if primary_emotion['score'] > 0.5:  # Only mention if strongly detected
        if primary_emotion['label'] == 'anger':
            summary_parts.append("Angry tone detected")
        elif primary_emotion['label'] == 'disgust':
            summary_parts.append("Disgust expressed in content")
        elif primary_emotion['label'] == 'fear':
            summary_parts.append("Fear-inducing language found")
        elif primary_emotion['label'] == 'sadness':
            summary_parts.append("Sad or depressing tone identified")
    
    # Combine all parts
    if not summary_parts:
        return "No concerning patterns detected in the text"
    
    # Create a more natural sounding summary
    if len(summary_parts) == 1:
        return summary_parts[0] + "."
    elif len(summary_parts) == 2:
        return summary_parts[0] + " and " + summary_parts[1].lower() + "."
    else:
        return ", ".join(summary_parts[:-1]) + ", and " + summary_parts[-1].lower() + "."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')
    result = detect_cyberbullying(text)
    return jsonify(result)

@app.route('/upload', methods=['POST'])
def upload_image():
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({
            "error": "No file uploaded",
            "verdict": "⚠️ Upload Error",
            "confidence_grade": "🔴 Critical"
        }), 400

    file = request.files['file']
    
    # Validate file selection
    if file.filename == '':
        return jsonify({
            "error": "No file selected",
            "verdict": "⚠️ Upload Error",
            "confidence_grade": "🔴 Critical"
        }), 400

    # Generate unique temp filename
    import uuid
    import os
    temp_id = uuid.uuid4().hex
    file_path = f"temp_{temp_id}.jpg"
    
    try:
        # Save uploaded file
        file.save(file_path)
        
        # OCR Processing
        ocr_result = process_image(file_path)
        if 'error' in ocr_result:
            return jsonify({
                **ocr_result,
                "verdict": "⚠️ OCR Error",
                "confidence_grade": "🔴 High"
            }), 400
            
        # Text Analysis
        analysis_result = detect_cyberbullying(ocr_result['extracted_text'])
        
        # Combine results
        return jsonify({
            **analysis_result,
            "extracted_text": ocr_result['extracted_text'],
            "ocr_raw": ocr_result.get('ocr_details', []),
            "processing_stage": "full_analysis"
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Processing error: {str(e)}",
            "verdict": "⚠️ System Error",
            "confidence_grade": "🔴 Critical"
        }), 500
        
    finally:
        # Cleanup temp file
        if os.path.exists(file_path):
            os.remove(file_path)



@app.route('/submit-complaint', methods=['POST'])
def handle_complaint():
    try:
        # Get form data
        incident_date = request.form.get('date')
        platform = request.form.get('platform')
        description = request.form.get('description')
        offensive_text = request.form.get('offensiveText')
        
        # Get analysis data
        analysis = json.loads(request.form.get('analysis', '{}'))
        text_analysis = analysis.get('text', {})
        image_analysis = analysis.get('image', {})

        # Process image file
        image_path = None
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file and image_file.filename != '':
                image_path = f"temp_{uuid.uuid4().hex}.jpg"
                image_file.save(image_path)

        # Generate PDF
        pdf_buffer = generate_pdf_report({
            'date': incident_date,
            'platform': platform,
            'description': description,
            'offensive_text': offensive_text,
            'text_analysis': text_analysis,
            'image_analysis': image_analysis
        }, image_path)

        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name='cyberbullying_report.pdf',
            mimetype='application/pdf'
        )

    except Exception as e:
        app.logger.error(f"Complaint error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)

def generate_pdf_report(data, image_path=None):
    """Generates PDF report for complaints"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Add metadata
    story.append(Paragraph("Cyberbullying Incident Report", styles['Title']))
    story.append(Spacer(1, 24))
    
    # Incident Details
    story.append(Paragraph("<b>Incident Details</b>", styles['Heading2']))
    story.append(Paragraph(f"Date: {data.get('date', 'N/A')}"))
    story.append(Paragraph(f"Platform: {data.get('platform', 'Unknown').capitalize()}"))
    story.append(Spacer(1, 12))
    
    # Description
    story.append(Paragraph("<b>User Description</b>", styles['Heading2']))
    story.append(Paragraph(data.get('description', 'No description provided'), styles['Normal']))
    story.append(Spacer(1, 24))
    
    # Offensive Content Section
    story.append(Paragraph("<b>Offensive Content Analysis</b>", styles['Heading2']))
    
    if data.get('offensive_text'):
        story.append(Paragraph("<i>Submitted Text:</i>", styles['Heading3']))
        story.append(Paragraph(data['offensive_text'], styles['Normal']))
        story.append(Spacer(1, 12))
    
    if data.get('text_analysis'):
        story.append(Paragraph("<i>Text Analysis Results:</i>", styles['Heading3']))
        story.append(Paragraph(f"Verdict: {data['text_analysis'].get('verdict', 'N/A')}"))
        story.append(Paragraph(f"Confidence: {data['text_analysis'].get('confidence_grade', 'N/A')}"))
        story.append(Spacer(1, 12))
    
    if image_path:
        story.append(Paragraph("<i>Uploaded Image Analysis:</i>", styles['Heading3']))
        try:
            story.append(Image(image_path, width=400, height=300))
        except:
            story.append(Paragraph("(Failed to load image)", styles['Italic']))
        
        if data.get('image_analysis'):
            story.append(Paragraph(f"Extracted Text: {data['image_analysis'].get('extracted_text', '')}"))
        story.append(Spacer(1, 12))
    
    doc.build(story)
    buffer.seek(0)
    return buffer


if __name__ == '__main__':
    app.run(debug=True)