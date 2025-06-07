#!/usr/bin/env python
"""
Cosmic Market Oracle Dashboard Runner

Main entry point for running the Streamlit dashboard application.
"""

import os
import sys
import subprocess
import threading
import time
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def start_websocket_server():
    """Start the WebSocket server in background."""
    try:
        from src.ui.websocket_server import WebSocketServer
        
        logger.info("Starting WebSocket server...")
        server = WebSocketServer(host="localhost", port=8765)
        server.start_server()
        
    except Exception as e:
        logger.error(f"Failed to start WebSocket server: {e}")
        logger.info("Dashboard will run without real-time features")

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'streamlit',
        'streamlit-authenticator', 
        'streamlit-option-menu',
        'plotly',
        'websockets',
        'redis'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info("Please install missing packages with:")
        logger.info(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def setup_environment():
    """Set up environment variables and directories."""
    # Create necessary directories
    directories = [
        "data",
        "data/users",
        "data/preferences", 
        "data/feedback",
        "data/sessions",
        "logs",
        "reports"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    # Set environment variables
    os.environ.setdefault("STREAMLIT_SERVER_PORT", "8501")
    os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

def run_streamlit_app():
    """Run the Streamlit application."""
    dashboard_path = project_root / "src" / "ui" / "dashboard.py"
    
    if not dashboard_path.exists():
        logger.error(f"Dashboard file not found: {dashboard_path}")
        logger.error("Please ensure src/ui/dashboard.py exists")
        return False
    
    try:
        # Run Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "false",
            "--theme.primaryColor", "#1e3c72",
            "--theme.backgroundColor", "#ffffff",
            "--theme.secondaryBackgroundColor", "#f0f2f6"
        ]
        
        logger.info("Starting Streamlit dashboard...")
        logger.info(f"Dashboard will be available at: http://localhost:8501")
        
        # Run the command
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start Streamlit: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
        return True
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

def main():
    """Main function to run the dashboard."""
    print("=" * 60)
    print("🌟 COSMIC MARKET ORACLE DASHBOARD 🌟")
    print("=" * 60)
    print()
    
    # Check dependencies
    logger.info("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Setup environment
    logger.info("Setting up environment...")
    setup_environment()
    
    # Start WebSocket server in background thread
    logger.info("Starting background services...")
    websocket_thread = threading.Thread(target=start_websocket_server, daemon=True)
    websocket_thread.start()
    
    # Give WebSocket server time to start
    time.sleep(2)
    
    # Display startup information
    print("🚀 Starting Cosmic Market Oracle Dashboard...")
    print()
    print("📊 Features Available:")
    print("  ✅ Interactive Market Analysis")
    print("  ✅ Real-time Astrological Insights") 
    print("  ✅ AI-Powered Predictions")
    print("  ✅ Trading Strategy Backtesting")
    print("  ✅ Live Alerts & Notifications")
    print("  ✅ User Authentication & Preferences")
    print()
    print("🌐 Access Information:")
    print("  📱 Web Dashboard: http://localhost:8501")
    print("  🔌 WebSocket Server: ws://localhost:8765")
    print()
    print("👤 Default Login:")
    print("  Username: admin")
    print("  Password: cosmic123")
    print("  (Change password after first login)")
    print()
    print("=" * 60)
    print()
    
    # Run the Streamlit app
    try:
        success = run_streamlit_app()
        if success:
            logger.info("Dashboard stopped successfully")
        else:
            logger.error("Dashboard stopped with errors")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        logger.info("Cleaning up...")
        print("\n🌟 Thank you for using Cosmic Market Oracle! 🌟")

if __name__ == "__main__":
    main()