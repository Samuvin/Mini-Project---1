#!/bin/bash

# Script to build and test Docker container

set -e

echo "=========================================="
echo "Building Docker Image"
echo "=========================================="
docker build -t parkinsons-prediction .

echo ""
echo "=========================================="
echo "Starting Docker Container"
echo "=========================================="
docker run -d \
  --name parkinsons-test \
  -p 8000:8000 \
  --env-file .env \
  parkinsons-prediction

echo ""
echo "Waiting for container to start..."
sleep 10

echo ""
echo "=========================================="
echo "Checking Container Status"
echo "=========================================="
docker ps | grep parkinsons-test

echo ""
echo "=========================================="
echo "Checking Health Endpoint"
echo "=========================================="
curl -f http://localhost:8000/api/health || echo "Health check failed"

echo ""
echo "=========================================="
echo "Container Logs (last 20 lines)"
echo "=========================================="
docker logs --tail 20 parkinsons-test

echo ""
echo "=========================================="
echo "To stop and remove container:"
echo "  docker stop parkinsons-test"
echo "  docker rm parkinsons-test"
echo "=========================================="
