#!/bin/bash

# Parkinson's Disease Prediction System - Stop Script
# Gracefully stops the running server

echo "============================================================"
echo "Parkinson's Disease Prediction System"
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

# Check if server is running (Waitress or python wsgi.py)
print_info "Looking for running server..."

PIDS=$(pgrep -f "waitress-serve.*wsgi:app" || pgrep -f "wsgi:app" || pgrep -f "wsgi.py" || true)

if [ -z "$PIDS" ]; then
    print_info "No server is currently running"
    echo ""
    exit 0
fi

# Stop the server
print_info "Stopping server (PIDs: $PIDS)..."
pkill -TERM -f "waitress-serve.*wsgi:app" 2>/dev/null
pkill -TERM -f "wsgi:app" 2>/dev/null
pkill -TERM -f "wsgi.py" 2>/dev/null

# Wait for graceful shutdown
sleep 2

REMAINING=$(pgrep -f "waitress-serve.*wsgi:app" || pgrep -f "wsgi:app" || pgrep -f "wsgi.py" || true)

if [ -z "$REMAINING" ]; then
    print_success "Server stopped successfully"
else
    print_info "Force stopping remaining processes..."
    pkill -KILL -f "waitress-serve.*wsgi:app" 2>/dev/null
    pkill -KILL -f "wsgi:app" 2>/dev/null
    pkill -KILL -f "wsgi.py" 2>/dev/null
    sleep 1
    print_success "Server stopped"
fi

echo ""
echo "============================================================"
print_success "Server is now stopped"
echo "To start again, run: ./start.sh"
echo "============================================================"
echo ""

