# Deployment Guide - Parkinson's Disease Detection System

## 🚀 Render Deployment

### Prerequisites
- GitHub account
- Render account (free tier works)
- Git repository for the project

### Quick Deploy to Render

#### Option 1: Using render.yaml (Recommended)

1. **Push to GitHub:**
```bash
git add .
git commit -m "Add Docker configuration for Render deployment"
git push origin main
```

2. **Connect to Render:**
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New +" → "Blueprint"
   - Connect your GitHub repository
   - Select the repository containing this project
   - Render will automatically detect `render.yaml`
   - Click "Apply" to deploy

3. **Wait for Build:**
   - First build takes 10-15 minutes (includes model training)
   - Subsequent builds are faster (~5 minutes)
   - Watch the logs for any errors

4. **Access Your App:**
   - Once deployed, Render provides a URL like: `https://parkinsons-detection.onrender.com`
   - Health check: `https://your-app.onrender.com/api/health`

#### Option 2: Manual Setup

1. **Create New Web Service:**
   - Go to Render Dashboard
   - Click "New +" → "Web Service"
   - Connect your repository

2. **Configure Service:**
   ```
   Name: parkinsons-detection
   Environment: Docker
   Region: Oregon (or closest to you)
   Branch: main
   ```

3. **Set Environment Variables:**
   ```
   FLASK_ENV=production
   PORT=8000
   PYTHONUNBUFFERED=1
   ```

4. **Deploy:**
   - Click "Create Web Service"
   - Wait for deployment to complete

### Important Notes

- **Free Tier Limitations:**
  - Service spins down after 15 minutes of inactivity
  - First request after spin-down takes 30-60 seconds
  - 750 hours/month of runtime

- **Build Process:**
  - Downloads speech dataset
  - Generates handwriting and gait datasets
  - Trains all three models (speech, handwriting, gait)
  - Total build time: ~10-15 minutes

- **Health Check:**
  - Endpoint: `/api/health`
  - Returns model status and loaded modalities

## 🐳 Local Docker Deployment

### Build and Run with Docker

```bash
# Build the Docker image
docker build -t parkinsons-detection .

# Run the container
docker run -p 8000:8000 parkinsons-detection

# Access the app
open http://localhost:8000
```

### Using Docker Compose

```bash
# Build and start
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## 📊 Monitoring

### Health Check Endpoint

```bash
curl https://your-app.onrender.com/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "models_loaded": ["speech", "handwriting", "gait"],
  "model_info": {
    "loaded_models": ["speech", "handwriting", "gait"],
    "model_details": {
      "speech": {...},
      "handwriting": {...},
      "gait": {...}
    }
  }
}
```

### Test Prediction

```bash
curl -X POST https://your-app.onrender.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "speech_features": [163.945, 85.185, 233.197, 0.00835, 0.00241, 0.148, 0.766, 0.00989, 0.0857, 0.0369, 0.0393, 0.0619, 0.592, 0.0983, 0.157, 0.570, 0.598, 0.529, 0.578, 1.004, 1.173, 0.0828]
  }'
```

## 🔧 Troubleshooting

### Build Fails

**Problem:** Docker build timeout
**Solution:** Increase build timeout in Render settings (Advanced → Build Command Timeout)

**Problem:** Model training fails
**Solution:** Check logs for dataset download errors. Ensure sufficient memory.

### Runtime Issues

**Problem:** 502 Bad Gateway
**Solution:** Service is starting up. Wait 30-60 seconds and retry.

**Problem:** "Model not loaded" error
**Solution:** Check `/api/health` endpoint. Models should be in `models_loaded` array.

### Memory Issues

**Problem:** Container crashes due to OOM
**Solution:** 
- Upgrade to paid Render plan (more memory)
- Or reduce workers in Dockerfile: `--workers 1`

## 📦 Alternative Deployment Options

### Heroku

1. Create `Procfile`:
```
web: gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 wsgi:app
```

2. Create `heroku.yml`:
```yaml
build:
  docker:
    web: Dockerfile
```

3. Deploy:
```bash
heroku create parkinsons-detection
git push heroku main
```

### Railway

1. Connect GitHub repository
2. Railway auto-detects Dockerfile
3. Deploy with one click

### AWS ECS / Google Cloud Run

Both support Docker containers. Follow their respective documentation for container deployment.

## 🔐 Production Checklist

- [ ] Set secure `SECRET_KEY` in environment variables
- [ ] Enable HTTPS (Render provides this automatically)
- [ ] Set up monitoring and alerts
- [ ] Configure custom domain (optional)
- [ ] Set up backup strategy for logs
- [ ] Review and adjust worker count based on traffic
- [ ] Set up error tracking (e.g., Sentry)
- [ ] Configure CORS if needed for API access

## 📈 Scaling

### Horizontal Scaling
- Increase number of instances in Render
- Use load balancer for distribution

### Vertical Scaling
- Upgrade Render plan for more CPU/RAM
- Increase worker count in Dockerfile

### Performance Optimization
- Enable Redis caching for predictions
- Use CDN for static assets
- Implement request rate limiting

## 🆘 Support

For issues:
1. Check Render logs: Dashboard → Your Service → Logs
2. Test locally with Docker first
3. Verify all environment variables are set
4. Check health endpoint status

## 📝 Notes

- Models are trained during Docker build (baked into image)
- No persistent storage needed for models
- Stateless application - easy to scale
- All dependencies included in container

