# Quick Start Guide 🚀

## Two Simple Commands

### Start the System
```bash
./start.sh
```

**What it does AUTOMATICALLY:**
- ✅ Checks Python installation
- ✅ Creates virtual environment (if needed)
- ✅ Installs all dependencies (if needed)
- ✅ **Downloads speech dataset** (if missing)
- ✅ Checks for handwriting & gait datasets
- ✅ **Trains ML model** (if missing or outdated)
- ✅ Stops any existing server
- ✅ Starts new server at http://localhost:8000

**First time run:** Takes 3-6 minutes (downloads data + trains model)  
**Subsequent runs:** Takes ~5 seconds (just starts server)  
**After dataset update:** Automatically retrains model

---

### Stop the System
```bash
./stop.sh
```

**What it does:**
- ✅ Gracefully stops the server
- ✅ Cleans up all processes

---

## That's It! 🎉

### Access the Application
Once started, open your browser to:
```
http://localhost:8000
```

### Features Available
- 🎤 **Speech Analysis** - Record voice or upload audio
- ✍️ **Handwriting Analysis** - Upload spiral/writing images  
- 🚶 **Gait Analysis** - Upload walking videos
- 📊 **AI Predictions** - Real-time Parkinson's detection
- 💡 **Example Files** - Try examples in each tab

---

## Troubleshooting

### Server won't start?
```bash
# Stop any conflicting processes
./stop.sh

# Try starting again
./start.sh
```

### Port 8000 already in use?
Edit `start.sh` and change `8000` to another port (e.g., `8080`)

### Dependencies failed to install?
Make sure you have Python 3.8+ installed:
```bash
python3 --version
```

---

## File Structure
```
fn/
├── start.sh          ← Start everything
├── stop.sh           ← Stop server
├── train.py          ← Train model (auto-run by start.sh)
├── models/           ← Trained models
├── data/             ← Datasets
└── webapp/           ← Web interface
    ├── templates/    ← HTML pages
    ├── static/       ← CSS/JS/Examples
    └── api/          ← Backend endpoints
```

---

## Development Mode

### View Logs
The server logs appear in the terminal where you ran `./start.sh`

### Restart After Code Changes
```bash
./stop.sh
./start.sh
```

### Check Server Status
```bash
curl http://localhost:8000/api/health
```

---

## Production Deployment

The system is production-ready:
- ✅ Gunicorn WSGI server (2 workers)
- ✅ Real ML models (no dummy data)
- ✅ Automatic setup & installation
- ✅ Graceful shutdown
- ✅ Error handling

---

## Need Help?

1. **Check the logs** - They appear in the terminal
2. **Try restarting** - `./stop.sh` then `./start.sh`
3. **Read the docs** - See `README.md` for details
4. **Test the examples** - Click "Use This Example" in any tab

---

**Made with ❤️ for Parkinson's Disease Early Detection**

**Version:** 2.0  
**Last Updated:** October 2025

