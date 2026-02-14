# Windows Setup Instructions

## Issue: Gunicorn Not Compatible with Windows

**Problem:** `gunicorn` requires the `fcntl` module which is Unix-only and doesn't exist on Windows. This causes `ModuleNotFoundError: No module named 'fcntl'`.

## Solutions

### Option 1: Use Windows Batch Script (Recommended for Windows)

Instead of `./start.sh`, use the Windows batch script:

```bash
start.bat
```

This script uses Flask's development server which works on Windows.

### Option 2: Use Docker (Best for Production)

Since you mentioned Docker in your setup instructions, **use Docker on Windows**:

1. **Start Docker Desktop** (Windows)
2. **Open Git Bash or PowerShell**
3. **Build and run with Docker:**

```bash
# Build the image
docker build -t parkinsons-prediction .

# Run the container
docker run -d -p 8000:8000 --env-file .env --name parkinsons-app parkinsons-prediction

# Check logs
docker logs parkinsons-app

# Access at http://localhost:8000
```

### Option 3: Use Flask Directly (Quick Test)

```bash
# Activate virtual environment
venv\Scripts\activate

# Set environment variables
set FLASK_APP=wsgi:app
set FLASK_ENV=development

# Run Flask
python -m flask run --host=0.0.0.0 --port=8000
```

## Updated Windows Setup Steps

1. **Start Docker Desktop** (if using Docker)
   - Or skip Docker and use native Windows

2. **Open Visual Studio Code**

3. **Clone Repository** (First time only)
   ```bash
   git clone https://github.com/Samuvin/Mini-Project---1.git
   ```

4. **Open Project Folder**
   - In VS Code: `File` → `Open Folder...`
   - Select the `Mini-Project---1` folder

5. **Start Application**
   - **Option A (Windows Native):** Run `start.bat` in PowerShell/CMD
   - **Option B (Docker):** Use Docker commands above
   - **Option C (Git Bash):** Use `start.bat` (works in Git Bash too)

6. **Access Application**
   - Open browser: `http://localhost:8000`

## Notes

- **`start.sh`** is for Linux/Mac (uses gunicorn)
- **`start.bat`** is for Windows (uses Flask dev server)
- **Docker** works on all platforms and uses gunicorn inside the container

## Troubleshooting

### "start.bat is not recognized"
- Make sure you're in the project directory
- Run: `.\start.bat` (with backslash)

### "Python not found"
- Install Python 3.8+ from python.org
- Make sure Python is in your PATH

### Port 8000 already in use
- Stop other applications using port 8000
- Or change port in `start.bat`: change `--port=8000` to another port
