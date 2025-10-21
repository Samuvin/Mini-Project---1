#!/bin/bash

# Parkinson's Disease Detection System - Stop Script
# Gracefully stops the running server

echo "============================================================"
echo "Parkinson's Disease Detection System"
echo "Stop Server"
echo "============================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if server is running
print_info "Looking for running server..."

# Find Gunicorn processes
PIDS=$(pgrep -f "gunicorn.*wsgi:app")

if [ -z "$PIDS" ]; then
    print_info "No server is currently running"
    echo ""
    exit 0
fi

# Stop the server
print_info "Stopping server (PIDs: $PIDS)..."
pkill -TERM -f "gunicorn.*wsgi:app"

# Wait for graceful shutdown
sleep 2

# Check if processes are still running
REMAINING=$(pgrep -f "gunicorn.*wsgi:app")

if [ -z "$REMAINING" ]; then
    print_success "Server stopped successfully"
else
    print_info "Force stopping remaining processes..."
    pkill -KILL -f "gunicorn.*wsgi:app"
    sleep 1
    print_success "Server stopped"
fi

echo ""
echo "============================================================"
print_success "Server is now stopped"
echo "To start again, run: ./start.sh"
echo "============================================================"
echo ""

