"""
WSGI entry point for Parkinson's Disease Detection System.
This file is used by WSGI servers like Gunicorn.
"""

from dotenv import load_dotenv

# Load environment variables from .env before anything else.
load_dotenv()

from webapp.app import create_app

# Create the Flask application instance
app = create_app()

if __name__ == "__main__":
    # Direct execution not recommended - use Gunicorn instead
    print("Please run the application using Gunicorn:")
    print("  gunicorn -c gunicorn_config.py wsgi:app")
    print("Or use the start script:")
    print("  ./start_server.sh")

