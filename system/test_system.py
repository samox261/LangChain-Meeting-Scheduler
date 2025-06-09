import os
import time
import logging
from .health_check import app as health_app
from .config_validator import ConfigValidator
from .state_manager import StateManager
import threading
import requests
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def start_health_check_server():
    """Start the health check server in a separate thread"""
    health_app.run(host='0.0.0.0', port=5000, use_reloader=False)

def test_health_endpoints():
    """Test the health check endpoints"""
    print("\n=== Testing Health Check Endpoints ===")
    
    # Wait for server to start
    time.sleep(2)
    
    try:
        # Test health endpoint
        health_response = requests.get('http://localhost:5000/health')
        print("\nHealth Status:")
        print(json.dumps(health_response.json(), indent=2))
        
        # Test metrics endpoint
        metrics_response = requests.get('http://localhost:5000/metrics')
        print("\nSystem Metrics:")
        print(json.dumps(metrics_response.json(), indent=2))
    except Exception as e:
        print(f"Error testing health endpoints: {e}")

def test_config_validator():
    """Test the configuration validator"""
    print("\n=== Testing Configuration Validator ===")
    
    validator = ConfigValidator()
    validation_report = validator.get_validation_report()
    print("\nConfiguration Validation Report:")
    print(validation_report)

def test_state_manager():
    """Test the state manager functionality"""
    print("\n=== Testing State Manager ===")
    
    manager = StateManager()
    
    # Test backup
    print("\nBacking up state files...")
    backup_success = manager.backup_state_files()
    print(f"Backup {'successful' if backup_success else 'failed'}")
    
    # Test cleanup
    print("\nCleaning up old backups...")
    cleanup_success = manager.cleanup_old_backups(days_to_keep=30)
    print(f"Cleanup {'successful' if cleanup_success else 'failed'}")
    
    # Test thread cleanup
    print("\nCleaning up inactive threads...")
    cleanup_results = manager.cleanup_inactive_threads(days_inactive=90)
    print("\nCleanup Results:")
    print(json.dumps(cleanup_results, indent=2))
    
    # Test validation
    print("\nValidating state files...")
    validation_results = manager.validate_state_files()
    print("\nValidation Results:")
    print(json.dumps(validation_results, indent=2))
    
    # Test statistics
    print("\nGetting state statistics...")
    stats = manager.get_state_stats()
    print("\nState Statistics:")
    print(json.dumps(stats, indent=2))

def main():
    print("Starting System Test Suite...")
    
    # Start health check server in a separate thread
    server_thread = threading.Thread(target=start_health_check_server)
    server_thread.daemon = True
    server_thread.start()
    
    try:
        # Run all tests
        test_health_endpoints()
        test_config_validator()
        test_state_manager()
        
        print("\nAll tests completed!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
    finally:
        # Keep the script running to allow health check server to continue
        print("\nPress Ctrl+C to stop the health check server...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping test suite...")

if __name__ == "__main__":
    main() 