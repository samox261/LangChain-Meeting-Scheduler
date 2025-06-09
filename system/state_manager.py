import os
import json
import logging
from datetime import datetime, timedelta
import pytz
from typing import Dict, Any, List, Optional
import shutil
from dateutil import parser as dateutil_parser

class StateManager:
    def __init__(self, backup_dir: str = "state_backups"):
        self.backup_dir = backup_dir
        self._ensure_backup_dir()

    def _ensure_backup_dir(self):
        """Ensure backup directory exists"""
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)

    def backup_state_files(self) -> bool:
        """Backup all state files"""
        try:
            state_files = [f for f in os.listdir('.') if f.startswith('scheduling_states_') and f.endswith('.json')]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            for state_file in state_files:
                backup_path = os.path.join(self.backup_dir, f"{state_file}.{timestamp}")
                shutil.copy2(state_file, backup_path)
                logging.info(f"Backed up {state_file} to {backup_path}")
            
            return True
        except Exception as e:
            logging.error(f"Failed to backup state files: {str(e)}")
            return False

    def cleanup_old_backups(self, days_to_keep: int = 30) -> bool:
        """Remove backups older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            for backup_file in os.listdir(self.backup_dir):
                if backup_file.endswith('.json'):
                    file_path = os.path.join(self.backup_dir, backup_file)
                    file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                    if file_time < cutoff_date:
                        os.remove(file_path)
                        logging.info(f"Removed old backup: {backup_file}")
            return True
        except Exception as e:
            logging.error(f"Failed to cleanup old backups: {str(e)}")
            return False

    def cleanup_inactive_threads(self, days_inactive: int = 180) -> Dict[str, Any]:
        """Remove inactive threads from state files"""
        try:
            state_files = [f for f in os.listdir('.') if f.startswith('scheduling_states_') and f.endswith('.json')]
            cleanup_stats = {
                "files_processed": 0,
                "threads_removed": 0,
                "errors": []
            }
            
            cutoff_date = datetime.now(pytz.UTC) - timedelta(days=days_inactive)
            
            for state_file in state_files:
                try:
                    with open(state_file, 'r') as f:
                        state_data = json.load(f)
                    
                    original_thread_count = len([k for k in state_data.keys() if k != "processed_command_email_ids"])
                    threads_to_remove = []
                    
                    for thread_id, thread_data in state_data.items():
                        if thread_id != "processed_command_email_ids":
                            # Get the most recent timestamp from history
                            if thread_data.get("history") and len(thread_data["history"]) > 0:
                                last_update = thread_data["history"][-1]["timestamp"]
                                # Parse the timestamp and ensure it's timezone-aware
                                try:
                                    last_updated = dateutil_parser.parse(last_update)
                                    if last_updated.tzinfo is None:
                                        last_updated = pytz.UTC.localize(last_updated)
                                    if last_updated < cutoff_date:
                                        threads_to_remove.append(thread_id)
                                except Exception as e:
                                    logging.warning(f"Could not parse timestamp {last_update} for thread {thread_id}: {e}")
                            else:
                                # If no history, consider it inactive
                                threads_to_remove.append(thread_id)
                    
                    for thread_id in threads_to_remove:
                        del state_data[thread_id]
                        cleanup_stats["threads_removed"] += 1
                    
                    if threads_to_remove:
                        with open(state_file, 'w') as f:
                            json.dump(state_data, f, indent=2)
                        logging.info(f"Removed {len(threads_to_remove)} inactive threads from {state_file}")
                    
                    cleanup_stats["files_processed"] += 1
                except Exception as e:
                    cleanup_stats["errors"].append(f"Error processing {state_file}: {str(e)}")
            
            return cleanup_stats
        except Exception as e:
            logging.error(f"Failed to cleanup inactive threads: {str(e)}")
            return {"error": str(e)}

    def validate_state_files(self) -> Dict[str, Any]:
        """Validate all state files for consistency"""
        try:
            state_files = [f for f in os.listdir('.') if f.startswith('scheduling_states_') and f.endswith('.json')]
            validation_results = {
                "files_checked": 0,
                "valid_files": 0,
                "invalid_files": 0,
                "errors": []
            }
            
            for state_file in state_files:
                try:
                    with open(state_file, 'r') as f:
                        state_data = json.load(f)
                    
                    # Basic structure validation
                    if not isinstance(state_data, dict):
                        raise ValueError("State data must be a dictionary")
                    
                    # Check for processed_command_email_ids
                    if "processed_command_email_ids" not in state_data:
                        raise ValueError("Missing processed_command_email_ids field")
                    if not isinstance(state_data["processed_command_email_ids"], list):
                        raise ValueError("processed_command_email_ids must be a list")
                    
                    # Check for required fields in each thread
                    for thread_id, thread_data in state_data.items():
                        if thread_id != "processed_command_email_ids":
                            if not isinstance(thread_data, dict):
                                raise ValueError(f"Thread data for {thread_id} must be a dictionary")
                            
                            required_fields = ["history", "processed_command_email_ids_in_thread"]
                            for field in required_fields:
                                if field not in thread_data:
                                    raise ValueError(f"Missing required field '{field}' in thread {thread_id}")
                    
                    validation_results["valid_files"] += 1
                except Exception as e:
                    validation_results["invalid_files"] += 1
                    validation_results["errors"].append(f"Error in {state_file}: {str(e)}")
                
                validation_results["files_checked"] += 1
            
            return validation_results
        except Exception as e:
            logging.error(f"Failed to validate state files: {str(e)}")
            return {"error": str(e)}

    def get_state_stats(self) -> Dict[str, Any]:
        """Get statistics about state files"""
        try:
            state_files = [f for f in os.listdir('.') if f.startswith('scheduling_states_') and f.endswith('.json')]
            stats = {
                "total_files": len(state_files),
                "total_threads": 0,
                "active_threads": 0,
                "total_meetings": 0,
                "meetings_by_status": {},
                "file_sizes": {}
            }
            
            now = datetime.now(pytz.UTC)
            for state_file in state_files:
                try:
                    file_size = os.path.getsize(state_file)
                    stats["file_sizes"][state_file] = file_size
                    
                    with open(state_file, 'r') as f:
                        state_data = json.load(f)
                    
                    thread_count = len([k for k in state_data.keys() if k != "processed_command_email_ids"])
                    stats["total_threads"] += thread_count
                    
                    # Count active threads (updated in last 24h)
                    for thread_id, thread_data in state_data.items():
                        if thread_id != "processed_command_email_ids":
                            if thread_data.get("history"):
                                last_update = thread_data["history"][-1]["timestamp"]
                                try:
                                    last_updated = dateutil_parser.parse(last_update)
                                    if last_updated.tzinfo is None:
                                        last_updated = pytz.UTC.localize(last_updated)
                                    if (now - last_updated).total_seconds() < 86400:  # 24 hours
                                        stats["active_threads"] += 1
                                except Exception as e:
                                    logging.warning(f"Could not parse timestamp {last_update} for thread {thread_id}: {e}")
                            
                            # Count meetings
                            if thread_data.get("last_scheduled_event"):
                                stats["total_meetings"] += 1
                                status = "scheduled"  # Since we have last_scheduled_event
                                stats["meetings_by_status"][status] = stats["meetings_by_status"].get(status, 0) + 1
                
                except Exception as e:
                    logging.error(f"Error processing {state_file} for stats: {str(e)}")
            
            return stats
        except Exception as e:
            logging.error(f"Failed to get state stats: {str(e)}")
            return {"error": str(e)}

if __name__ == "__main__":
    # Example usage
    manager = StateManager()
    
    # Backup state files
    manager.backup_state_files()
    
    # Cleanup old backups (keep for 90 days instead of 30)
    manager.cleanup_old_backups(days_to_keep=90)
    
    # Cleanup inactive threads (180 days instead of 90)
    cleanup_results = manager.cleanup_inactive_threads(days_inactive=180)
    print("Cleanup Results:", cleanup_results)
    
    # Validate state files
    validation_results = manager.validate_state_files()
    print("Validation Results:", validation_results)
    
    # Get state statistics
    stats = manager.get_state_stats()
    print("State Statistics:", stats)
    
    # Check for empty state files
    for file_name, size in stats["file_sizes"].items():
        if size < 100:  # Less than 100 bytes
            print(f"\nWARNING: State file {file_name} is very small ({size} bytes). It may have been emptied by cleanup.")
            print("Consider restoring from backup if this was not intended.") 