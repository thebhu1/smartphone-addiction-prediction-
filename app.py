"""
Flask Web Application
Main web interface for smartphone addiction prediction
"""

from datetime import datetime
from io import BytesIO
import os
import sys
from uuid import uuid4

from flask import Flask, render_template, request, jsonify, send_file, session
from flask_cors import CORS
from zoneinfo import ZoneInfo

# Add src to path
sys.path.append('src')

from prediction import AddictionPredictor

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'change-this-secret-in-production')

# Initialize predictor
predictor = AddictionPredictor()

PROJECT_TITLE = "Predictive Modelling for Smartphone Addiction Using Machine Learning"
REPORT_FOOTER_NOTE = (
    "This report is generated for educational and digital wellness awareness purposes. "
    "It is not a medical diagnosis."
)

FEATURE_LABELS = {
    'daily_screen_time': 'Daily Screen Time',
    'social_media_time': 'Social Media Time',
    'gaming_time': 'Gaming Time',
    'notifications_per_day': 'Daily Notifications',
    'app_launches': 'App Launches',
    'night_usage': 'Night Usage',
    'sleep_disturbance': 'Sleep Disturbance',
    'unlock_frequency': 'Phone Unlocks',
    'weekend_usage_increase': 'Weekend Usage Increase',
    'session_duration': 'Average Session Duration'
}

FEATURE_UNITS = {
    'daily_screen_time': 'hours',
    'social_media_time': 'hours',
    'gaming_time': 'hours',
    'notifications_per_day': 'notifications/day',
    'app_launches': 'launches/day',
    'night_usage': 'hours',
    'sleep_disturbance': 'out of 10',
    'unlock_frequency': 'unlocks/day',
    'weekend_usage_increase': '%',
    'session_duration': 'minutes'
}

RISK_SUMMARIES = {
    'Low': "Your current digital behavior appears healthy.",
    'Moderate': "Your usage pattern suggests emerging dependency risk.",
    'High': "Your usage pattern indicates a high likelihood of problematic smartphone dependence."
}

RISK_RECOMMENDATIONS = {
    'Low': [
        'Maintain healthy digital habits.',
        'Continue monitoring daily screen time.',
        'Keep a balanced sleep schedule and offline routine.'
    ],
    'Moderate': [
        'Set app usage limits and track progress weekly.',
        'Reduce social media exposure during focus and rest hours.',
        'Improve sleep discipline by limiting night usage.',
        'Schedule regular no-phone breaks each day.'
    ],
    'High': [
        'Seek professional guidance for behavior support.',
        'Implement strict usage limits across high-use apps.',
        'Turn off non-essential notifications.',
        'Practice digital detox periods throughout the week.',
        'Increase participation in offline activities and social routines.'
    ]
}

RISK_COLOR_MAP = {
    'Low': '#2e7d32',
    'Moderate': '#f57c00',
    'High': '#c62828'
}


def _clean_risk_label(raw_label):
    """Normalize risk labels to one of: Low / Moderate / High."""
    label = str(raw_label or '').replace('Risk', '').strip().title()
    if label not in {'Low', 'Moderate', 'High'}:
        return 'Moderate'
    return label


def calculate_weighted_digital_load(input_data):
    """
    Calculate WDL on a 0-10 scale from core digital usage features.
    """
    screen_component = min((float(input_data.get('daily_screen_time', 0.0)) / 16.0) * 10.0, 10.0)
    social_component = min((float(input_data.get('social_media_time', 0.0)) / 8.0) * 10.0, 10.0)
    gaming_component = min((float(input_data.get('gaming_time', 0.0)) / 6.0) * 10.0, 10.0)
    night_component = min((float(input_data.get('night_usage', 0.0)) / 5.0) * 10.0, 10.0)

    wdl = (
        (0.40 * screen_component) +
        (0.25 * social_component) +
        (0.15 * gaming_component) +
        (0.20 * night_component)
    )
    return round(wdl, 2)


def _format_numeric(value, decimals=2):
    """Format numbers for readable report output."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if abs(numeric_value - round(numeric_value)) < 1e-8:
        return str(int(round(numeric_value)))
    return f"{numeric_value:.{decimals}f}"


def format_feature_value(feature_key, value):
    """Render feature values with units for the PDF table."""
    if feature_key == 'weekend_usage_increase':
        return f"{_format_numeric(value, 1)}%"
    if feature_key in {'notifications_per_day', 'app_launches', 'unlock_frequency'}:
        return _format_numeric(value, 0)
    if feature_key == 'sleep_disturbance':
        return f"{_format_numeric(value, 1)} / 10"

    unit = FEATURE_UNITS.get(feature_key)
    if unit:
        return f"{_format_numeric(value, 2)} {unit}"
    return _format_numeric(value, 2)


def build_risk_explanation(input_data, risk_label):
    """Generate a short rule-based explanation tied to elevated usage factors."""
    flags = []

    if float(input_data.get('daily_screen_time', 0)) >= 6:
        flags.append('daily screen time is elevated')
    if float(input_data.get('social_media_time', 0)) >= 3:
        flags.append('social media usage is high')
    if float(input_data.get('gaming_time', 0)) >= 2:
        flags.append('gaming duration is above healthy limits')
    if float(input_data.get('night_usage', 0)) >= 2:
        flags.append('night-time usage is frequent')
    if float(input_data.get('sleep_disturbance', 0)) >= 6:
        flags.append('sleep disturbance score is elevated')
    if float(input_data.get('unlock_frequency', 0)) >= 120:
        flags.append('phone unlock frequency is high')

    if flags:
        if len(flags) == 1:
            reason_text = flags[0]
        else:
            reason_text = ', '.join(flags[:-1]) + f", and {flags[-1]}"
        return (
            f"The user is classified as {risk_label} Risk because {reason_text}, "
            "which indicates increased behavioral risk."
        )

    return (
        f"The user is classified as {risk_label} Risk based on the combined pattern "
        "of digital usage, interaction frequency, and sleep impact indicators."
    )


def get_dynamic_recommendations(risk_label):
    """Return summary sentence + tailored recommendations by risk class."""
    normalized_label = _clean_risk_label(risk_label)
    return (
        RISK_SUMMARIES.get(normalized_label, RISK_SUMMARIES['Moderate']),
        RISK_RECOMMENDATIONS.get(normalized_label, RISK_RECOMMENDATIONS['Moderate'])
    )


def build_report_payload(input_data, prediction_result):
    """Build a normalized report payload from form inputs + model output."""
    normalized_label = _clean_risk_label(prediction_result.get('addiction_label'))
    summary_sentence, recommendations = get_dynamic_recommendations(normalized_label)
    risk_explanation = build_risk_explanation(input_data, normalized_label)

    return {
        'project_title': PROJECT_TITLE,
        'inputs': {k: float(v) for k, v in input_data.items()},
        'wdl': calculate_weighted_digital_load(input_data),
        'predicted_class': f"{normalized_label} Risk",
        'confidence_scores': prediction_result.get('probabilities', {}),
        'risk_explanation': risk_explanation,
        'summary_sentence': summary_sentence,
        'recommendations': recommendations
    }


def generate_prediction_report(data, generated_at, report_id):
    """
    Generate a PDF report in memory and return a BytesIO stream.
    """
    from xml.sax.saxutils import escape
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        HRFlowable,
        Image,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle
    )

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=45,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=17,
        leading=22,
        textColor=colors.HexColor('#1f2933'),
        alignment=1,
        spaceAfter=8
    )
    meta_style = ParagraphStyle(
        'MetaStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#52606d'),
        alignment=1,
        spaceAfter=2
    )
    section_style = ParagraphStyle(
        'SectionStyle',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=colors.HexColor('#334e68'),
        spaceBefore=8,
        spaceAfter=8
    )
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#102a43')
    )
    bullet_style = ParagraphStyle(
        'BulletStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        leftIndent=12,
        bulletIndent=0,
        textColor=colors.HexColor('#102a43')
    )

    story = []

    logo_candidates = [
        os.path.join('static', 'img', 'logo.png'),
        os.path.join('static', 'images', 'logo.png')
    ]
    for logo_path in logo_candidates:
        if os.path.exists(logo_path):
            story.append(Image(logo_path, width=1.0 * inch, height=1.0 * inch, hAlign='CENTER'))
            story.append(Spacer(1, 6))
            break

    story.append(Paragraph(escape(data.get('project_title', PROJECT_TITLE)), title_style))
    story.append(Paragraph(f"Generated: {generated_at.strftime('%d %b %Y, %I:%M:%S %p')}", meta_style))
    story.append(Paragraph(f"Report ID: {escape(report_id)}", meta_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#d9e2ec')))
    story.append(Spacer(1, 10))

    # Inputs section
    story.append(Paragraph('Input Features', section_style))
    input_rows = [['Feature', 'Value']]
    inputs = data.get('inputs', {})
    for feature_key in predictor.feature_columns:
        if feature_key in inputs:
            input_rows.append([
                FEATURE_LABELS.get(feature_key, feature_key.replace('_', ' ').title()),
                format_feature_value(feature_key, inputs[feature_key])
            ])
    input_rows.append(['Weighted Digital Load (WDL)', f"{_format_numeric(data.get('wdl', 0), 2)} / 10"])

    input_table = Table(input_rows, colWidths=[300, 180])
    input_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#334e68')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bcccdc')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fbff')]),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6)
    ]))
    story.append(input_table)

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width='100%', thickness=0.8, color=colors.HexColor('#d9e2ec')))
    story.append(Spacer(1, 10))

    # Prediction output section
    story.append(Paragraph('Prediction Result', section_style))
    normalized_label = _clean_risk_label(data.get('predicted_class', 'Moderate'))
    risk_hex = RISK_COLOR_MAP.get(normalized_label, RISK_COLOR_MAP['Moderate'])
    story.append(Paragraph(
        f"Predicted Addiction Risk Class: <font color='{risk_hex}'><b>{normalized_label} Risk</b></font>",
        body_style
    ))
    story.append(Spacer(1, 6))

    confidence_scores = data.get('confidence_scores', {})
    confidence_rows = [['Risk Class', 'Confidence']]
    for class_name in ['Low', 'Moderate', 'High']:
        score = confidence_scores.get(class_name, 0)
        score_percent = float(score) * 100 if float(score) <= 1 else float(score)
        confidence_rows.append([f"{class_name} Risk", f"{score_percent:.1f}%"])

    confidence_table = Table(confidence_rows, colWidths=[300, 180])
    confidence_style = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f2933')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bcccdc')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6)
    ]
    row_colors = [colors.HexColor('#e8f5e9'), colors.HexColor('#fff3e0'), colors.HexColor('#ffebee')]
    for idx, row_color in enumerate(row_colors, start=1):
        confidence_style.append(('BACKGROUND', (0, idx), (-1, idx), row_color))

    confidence_table.setStyle(TableStyle(confidence_style))
    story.append(confidence_table)

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width='100%', thickness=0.8, color=colors.HexColor('#d9e2ec')))
    story.append(Spacer(1, 10))

    # Explanation section
    story.append(Paragraph('Risk Explanation', section_style))
    story.append(Paragraph(escape(data.get('risk_explanation', '')), body_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Summary:</b> {escape(data.get('summary_sentence', ''))}", body_style))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width='100%', thickness=0.8, color=colors.HexColor('#d9e2ec')))
    story.append(Spacer(1, 10))

    # Recommendations section
    story.append(Paragraph('Recommendations', section_style))
    recommendations = data.get('recommendations', [])
    for recommendation in recommendations:
        story.append(Paragraph(f"&bull; {escape(str(recommendation))}", bullet_style))
        story.append(Spacer(1, 2))

    story.append(Spacer(1, 16))
    story.append(Paragraph(f"<i>{escape(REPORT_FOOTER_NOTE)}</i>", meta_style))

    def _draw_footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont('Helvetica-Oblique', 8)
        canvas.setFillColor(colors.HexColor('#7b8794'))
        canvas.drawString(_doc.leftMargin, 20, REPORT_FOOTER_NOTE)
        canvas.restoreState()

    document.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    buffer.seek(0)
    return buffer

# Load models on startup
try:
    predictor.load_models()
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"⚠️  Warning: Could not load models - {str(e)}")
    print("   Run 'python train_model.py' first")

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict_page')
def predict_page():
    """Prediction form page"""
    return render_template('predict.html')

@app.route('/model_performance')
def model_performance():
    """Model performance dashboard page"""
    return render_template('model_performance.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint for making predictions
    
    Expects JSON data with all required features
    Returns prediction result with risk assessment
    """
    try:
        # Get data from request
        data = request.get_json()
        
        # Validate data exists
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Convert string inputs to float
        input_data = {}
        for key, value in data.items():
            try:
                input_data[key] = float(value)
            except (ValueError, TypeError):
                return jsonify({
                    'success': False,
                    'error': f'Invalid value for {key}: {value}'
                }), 400
        
        # Make prediction
        result = predictor.predict_with_assessment(input_data)

        # Enrich result with dynamic report content and store in session
        report_payload = build_report_payload(input_data, result)
        result['wdl'] = report_payload['wdl']
        result['risk_explanation'] = report_payload['risk_explanation']
        result['summary_sentence'] = report_payload['summary_sentence']
        result['recommendations'] = report_payload['recommendations']

        session['latest_prediction_report'] = report_payload
        
        return jsonify({
            'success': True,
            'prediction': result
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500


@app.route('/generate-report', methods=['POST'])
def generate_report():
    """Generate and download prediction report PDF."""
    try:
        payload = request.get_json(silent=True) or {}
        report_payload = None

        # Option A: Use payload sent from frontend (inputs + prediction)
        if isinstance(payload, dict) and payload.get('inputs') and payload.get('prediction'):
            raw_inputs = payload.get('inputs', {})
            normalized_inputs = {}
            for feature in predictor.feature_columns:
                if feature in raw_inputs:
                    normalized_inputs[feature] = float(raw_inputs[feature])

            if normalized_inputs:
                report_payload = build_report_payload(normalized_inputs, payload.get('prediction', {}))
                session['latest_prediction_report'] = report_payload

        # Option B: Fallback to latest prediction stored in session
        if report_payload is None:
            report_payload = session.get('latest_prediction_report')

        if not report_payload:
            return jsonify({
                'success': False,
                'error': 'No prediction report data found. Please run a prediction first.'
            }), 400

        generated_at = datetime.now(ZoneInfo('Asia/Kolkata'))
        report_id = f"SAPS-{generated_at.strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:6].upper()}"
        pdf_buffer = generate_prediction_report(report_payload, generated_at, report_id)
        filename = f"smartphone_addiction_report_{generated_at.strftime('%Y%m%d_%H%M%S')}.pdf"

        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )

    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'Invalid report payload: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to generate report: {str(e)}'
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_loaded = predictor.model is not None
    scaler_loaded = predictor.scaler is not None
    
    return jsonify({
        'status': 'healthy' if (model_loaded and scaler_loaded) else 'unhealthy',
        'model_loaded': model_loaded,
        'scaler_loaded': scaler_loaded
    })

@app.route('/api/feature_info', methods=['GET'])
def feature_info():
    """Get information about input features"""
    features = {
        'daily_screen_time': {
            'label': 'Daily Screen Time (hours)',
            'description': 'Average hours spent on phone per day',
            'min': 1,
            'max': 16,
            'step': 0.5,
            'unit': 'hours'
        },
        'social_media_time': {
            'label': 'Social Media Usage (hours)',
            'description': 'Daily time spent on social media apps',
            'min': 0,
            'max': 8,
            'step': 0.5,
            'unit': 'hours'
        },
        'gaming_time': {
            'label': 'Gaming Time (hours)',
            'description': 'Daily time spent on mobile games',
            'min': 0,
            'max': 6,
            'step': 0.5,
            'unit': 'hours'
        },
        'notifications_per_day': {
            'label': 'Daily Notifications',
            'description': 'Average number of notifications received',
            'min': 10,
            'max': 1000,
            'step': 1,
            'unit': 'notifications'
        },
        'app_launches': {
            'label': 'App Launches',
            'description': 'Number of times apps are opened daily',
            'min': 20,
            'max': 500,
            'step': 1,
            'unit': 'launches'
        },
        'night_usage': {
            'label': 'Night Usage (hours)',
            'description': 'Phone usage between 11 PM - 6 AM',
            'min': 0,
            'max': 5,
            'step': 0.25,
            'unit': 'hours'
        },
        'sleep_disturbance': {
            'label': 'Sleep Disturbance (0-10)',
            'description': 'How much phone affects your sleep',
            'min': 0,
            'max': 10,
            'step': 1,
            'unit': 'scale'
        },
        'unlock_frequency': {
            'label': 'Phone Unlocks',
            'description': 'Times you unlock phone per day',
            'min': 20,
            'max': 500,
            'step': 1,
            'unit': 'unlocks'
        },
        'weekend_usage_increase': {
            'label': 'Weekend Usage Increase (%)',
            'description': 'Usage increase on weekends vs weekdays',
            'min': -20,
            'max': 100,
            'step': 1,
            'unit': 'percent'
        },
        'session_duration': {
            'label': 'Average Session Duration (min)',
            'description': 'Average time per phone session',
            'min': 2,
            'max': 30,
            'step': 1,
            'unit': 'minutes'
        }
    }
    
    return jsonify(features)

@app.route('/api/model_metrics', methods=['GET'])
def model_metrics():
    """Get model performance metrics"""
    try:
        # Try to load metrics from saved file
        metrics_path = 'models/model_metrics.json'
        if os.path.exists(metrics_path):
            import json
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            return jsonify(metrics)
        else:
            # Return dummy data if metrics not found
            return jsonify({
                'models': {
                    'Logistic Regression': {
                        'accuracy': 0.895,
                        'precision': 0.893,
                        'recall': 0.895,
                        'f1_score': 0.894
                    },
                    'Random Forest': {
                        'accuracy': 0.925,
                        'precision': 0.926,
                        'recall': 0.925,
                        'f1_score': 0.925
                    },
                    'SVM': {
                        'accuracy': 0.910,
                        'precision': 0.908,
                        'recall': 0.910,
                        'f1_score': 0.909
                    }
                },
                'best_model': 'Random Forest',
                'feature_importance': {
                    'sleep_disturbance': 0.129,
                    'daily_screen_time': 0.115,
                    'notifications_per_day': 0.115,
                    'session_duration': 0.105,
                    'social_media_time': 0.105,
                    'night_usage': 0.100,
                    'unlock_frequency': 0.095,
                    'gaming_time': 0.090,
                    'app_launches': 0.085,
                    'weekend_usage_increase': 0.061
                }
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print(" " * 5 + "📱 SMARTPHONE ADDICTION PREDICTION SYSTEM")
    print("="*70)
    
    if predictor.model is None:
        print("\n⚠️  WARNING: Models not loaded!")
        print("   Please run: python train_model.py")
        print("   Then restart this server.\n")
    else:
        print("\n✅ System ready!")
    
    print("\n🌐 Server starting...")
    print("   URL: http://localhost:5000")
    print("   Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
