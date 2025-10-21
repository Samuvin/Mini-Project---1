"""Flask web application for Parkinson's Disease detection."""

import sys
import os
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
import numpy as np

from src.utils.config import Config
from webapp.api.predict import predict_bp, load_model


def setup_logging(app):
    """Configure production logging."""
    if not app.debug:
        # Create logs directory
        logs_dir = project_root / 'logs'
        logs_dir.mkdir(exist_ok=True)
        
        # File handler
        file_handler = RotatingFileHandler(
            logs_dir / 'app.log',
            maxBytes=10240000,  # 10MB
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        
        app.logger.setLevel(logging.INFO)
        app.logger.info('Parkinson\'s Detection System startup')


def create_app(config_path=None):
    """
    Create and configure the Flask application.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Load configuration
    config = Config(config_path)
    webapp_config = config.get_webapp_config()
    
    # Application configuration
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB for video uploads
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24).hex())
    app.config['DEBUG'] = False
    app.config['TESTING'] = False
    
    # Setup logging
    setup_logging(app)
    
    # Enable CORS
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(predict_bp, url_prefix='/api')
    
    # Import and register file upload blueprint
    from webapp.api.file_upload import upload_bp
    app.register_blueprint(upload_bp, url_prefix='/api/upload')
    
    # Load model on startup
    try:
        load_model()
        print("Model loaded successfully on startup")
    except Exception as e:
        print(f"Warning: Could not load model on startup: {e}")
    
    # Routes
    @app.route('/')
    def index():
        """Home page."""
        return render_template('index.html')
    
    @app.route('/predict_page')
    def predict_page():
        """Prediction page."""
        return render_template('predict.html')
    
    @app.route('/about')
    def about():
        """About page."""
        return render_template('about.html')
    
    @app.route('/documentation')
    def documentation():
        """Documentation page."""
        return render_template('documentation.html')
    
    @app.route('/performance')
    def performance():
        """Model performance page."""
        from pathlib import Path
        import json
        
        # Get models directory
        models_dir = Path(app.root_path).parent / 'models'
        
        # Load metrics from JSON file
        metrics_file = models_dir / 'model_metrics.json'
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
        else:
            # Default metrics
            metrics = {
                'svm': {
                    'accuracy': 0.90,
                    'precision': 1.00,
                    'recall': 0.87,
                    'f1_score': 0.93,
                    'roc_auc': 0.95,
                    'available': False
                }
            }
        
        # Check available files
        available_files = {
            'svm_roc': (models_dir / 'svm_roc_curve.png').exists(),
            'lr_roc': (models_dir / 'lr_roc_curve.png').exists(),
            'svm_cm': (models_dir / 'svm_confusion_matrix.png').exists(),
            'lr_cm': (models_dir / 'lr_confusion_matrix.png').exists(),
        }
        
        return render_template('performance.html', 
                             metrics=metrics, 
                             files=available_files)
    
    @app.route('/model_images/<path:filename>')
    def model_images(filename):
        """Serve model performance images."""
        from flask import send_from_directory
        import os
        models_dir = os.path.join(app.root_path, '..', 'models')
        return send_from_directory(models_dir, filename)
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors."""
        return render_template('error.html', error_code=404, error_message='Page not found'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors."""
        return render_template('error.html', error_code=500, error_message='Internal server error'), 500
    
    @app.errorhandler(413)
    def too_large(error):
        """Handle file too large errors."""
        return jsonify({'error': 'File too large', 'success': False}), 413
    
    return app


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Parkinson's Disease Detection System")
    print("="*60)
    print("\nThis application should be run using Gunicorn:")
    print("  gunicorn -c gunicorn_config.py wsgi:app")
    print("\nOr use the start script:")
    print("  ./start_production.sh")
    print("="*60 + "\n")

