from flask import Flask, jsonify
import os
import json
import logging
from datetime import datetime
import pytz
from tools.calendar_tools import calendar_event_creator_tool
from tools.email_tools import email_reader_tool

app = Flask(__name__)

def check_calendar_api():
    try:
        # Try to list events for next 24 hours
        test_result = calendar_event_creator_tool.invoke({
            "summary": "Health Check",
            "start_datetime_iso": datetime.now().isoformat(),
            "end_datetime_iso": datetime.now().isoformat(),
            "attendees": [],
            "description": "System health check",
            "location": "Virtual",
            "timezone": "UTC"
        })
        return {"status": "healthy", "message": "Calendar API is accessible"}
    except Exception as e:
        return {"status": "unhealthy", "message": f"Calendar API error: {str(e)}"}

def check_email_api():
    try:
        # Try to read last 1 email
        test_result = email_reader_tool.invoke("1")
        return {"status": "healthy", "message": "Email API is accessible"}
    except Exception as e:
        return {"status": "unhealthy", "message": f"Email API error: {str(e)}"}

def check_state_files():
    try:
        state_files = [f for f in os.listdir('.') if f.startswith('scheduling_states_') and f.endswith('.json')]
        state_status = []
        for state_file in state_files:
            try:
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                state_status.append({
                    "file": state_file,
                    "status": "healthy",
                    "last_updated": state_data.get("last_updated", "unknown")
                })
            except Exception as e:
                state_status.append({
                    "file": state_file,
                    "status": "unhealthy",
                    "error": str(e)
                })
        return state_status
    except Exception as e:
        return [{"status": "unhealthy", "error": f"State files check error: {str(e)}"}]

@app.route('/')
def root():
    return jsonify({
        "service": "Meeting Scheduler Health Check",
        "endpoints": {
            "/health": "Get system health status",
            "/metrics": "Get system metrics"
        },
        "status": "running"
    })

@app.route('/health')
def health_check():
    calendar_status = check_calendar_api()
    email_status = check_email_api()
    state_status = check_state_files()
    
    overall_status = "healthy"
    if any(s.get("status") == "unhealthy" for s in [calendar_status, email_status] + state_status):
        overall_status = "unhealthy"
    
    return jsonify({
        "status": overall_status,
        "timestamp": datetime.now(pytz.UTC).isoformat(),
        "components": {
            "calendar_api": calendar_status,
            "email_api": email_status,
            "state_files": state_status
        }
    })

@app.route('/metrics')
def metrics():
    try:
        state_files = [f for f in os.listdir('.') if f.startswith('scheduling_states_') and f.endswith('.json')]
        metrics_data = {
            "total_threads": 0,
            "active_threads": 0,
            "total_meetings_scheduled": 0,
            "meetings_by_status": {},
            "last_24h_activity": 0
        }
        
        for state_file in state_files:
            with open(state_file, 'r') as f:
                state_data = json.load(f)
                
            # Count threads (excluding processed_command_email_ids)
            thread_count = len([k for k in state_data.keys() if k != "processed_command_email_ids"])
            metrics_data["total_threads"] += thread_count
            
            # Count active threads and meetings
            now = datetime.now(pytz.UTC)
            logging.info(f"Current time (UTC): {now.isoformat()}")
            active_count = 0
            
            for thread_id, thread_data in state_data.items():
                if thread_id != "processed_command_email_ids":
                    # Check if thread has history
                    if "history" in thread_data and thread_data["history"]:
                        try:
                            # Get the most recent timestamp from history
                            timestamps = []
                            for h in thread_data["history"]:
                                # Parse the timestamp and make it timezone-aware
                                ts_str = h["timestamp"]
                                logging.info(f"Processing timestamp: {ts_str}")
                                # Add timezone info if missing (assuming local timezone)
                                if not any(c in ts_str for c in ['+', '-', 'Z']):
                                    ts_str += '+03:00'  # Using +03:00 as it matches your event times
                                logging.info(f"Timestamp with timezone: {ts_str}")
                                parsed_ts = datetime.fromisoformat(ts_str)
                                # Make sure the timestamp is timezone-aware
                                if parsed_ts.tzinfo is None:
                                    parsed_ts = pytz.timezone('Asia/Beirut').localize(parsed_ts)
                                logging.info(f"Parsed timestamp: {parsed_ts.isoformat()}")
                                timestamps.append(parsed_ts)
                            
                            latest_timestamp = max(timestamps)
                            logging.info(f"Latest timestamp: {latest_timestamp.isoformat()}")
                            time_diff = (now - latest_timestamp).total_seconds()
                            logging.info(f"Time difference in seconds: {time_diff}")
                            
                            if time_diff < 86400:  # 24 hours
                                active_count += 1
                                metrics_data["last_24h_activity"] += 1
                                logging.info(f"Thread {thread_id} is active")
                            else:
                                logging.info(f"Thread {thread_id} is not active (too old)")
                        except (ValueError, TypeError) as e:
                            logging.error(f"Error parsing timestamp in thread {thread_id}: {e}")
                            continue
                    
                    # Count scheduled meetings - check if last_scheduled_event exists and has an eventId
                    if "last_scheduled_event" in thread_data and thread_data["last_scheduled_event"].get("eventId"):
                        metrics_data["total_meetings_scheduled"] += 1
                        # All meetings are considered "scheduled" if they have last_scheduled_event
                        status = "scheduled"
                        metrics_data["meetings_by_status"][status] = metrics_data["meetings_by_status"].get(status, 0) + 1
                        logging.info(f"Found scheduled meeting in thread {thread_id}")
            
            metrics_data["active_threads"] += active_count
        
        logging.info(f"Final metrics: {json.dumps(metrics_data, indent=2)}")
        return jsonify(metrics_data)
    except Exception as e:
        logging.error(f"Error in metrics endpoint: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 