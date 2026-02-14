"""WSGI entry point for Parkinson's Disease Prediction System.

This file is used by WSGI servers like Gunicorn.
"""

from dotenv import load_dotenv

# Load environment variables from .env before anything else.
load_dotenv()

from webapp.app import create_app

# Create the Flask application instance
app = create_app()

if __name__ == "__main__":
    # For Windows compatibility, allow direct Flask run
    import sys
    import os
    
    # Check if running on Windows
    if sys.platform == "win32":
        print("Running Flask development server (Windows)...")
        print("For production on Linux/Mac, use: gunicorn -c gunicorn_config.py wsgi:app")
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False)
    else:
        print("Please run the application using Gunicorn:")
        print("  gunicorn -c gunicorn_config.py wsgi:app")
        print("Or use the start script:")
        print("  ./start.sh")
