import os
import json
import yaml 
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pytz 
from dateutil import parser as dateutil_parser
from typing import Optional, Dict, Any, List # Ensure all are imported
import re # For parsing email from sender string

# Import your LangChain tools
from tools.email_tools import email_reader_tool
from tools.nlp_tools import email_analyzer_tool, normalize_datetime_with_llm # Assuming normalize_datetime_with_llm is in nlp_tools.py
from tools.calendar_tools import calendar_event_creator_tool

CONFIG_FILE = "config.yaml"

def load_config() -> Dict[str, Any]:
    """Loads the YAML configuration file."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
        if not config:
            print(f"Warning: {CONFIG_FILE} is empty. Using default values or expecting environment variables.")
            return {}
        return config
    except FileNotFoundError:
        print(f"Warning: {CONFIG_FILE} not found. Using default values or expecting environment variables.")
        return {}
    except Exception as e:
        print(f"Error loading {CONFIG_FILE}: {e}")
        return {}

def parse_datetime_from_llm(
    natural_date_time_str: str, 
    user_timezone_str: str, 
    reference_datetime: Optional[datetime] = None
) -> Optional[datetime]:
    """
    Uses an LLM to normalize a natural language date/time string to ISO format,
    then parses it into a timezone-aware datetime object.
    """
    if not natural_date_time_str:
        return None

    user_tz = pytz.timezone(user_timezone_str)
    if reference_datetime is None:
        reference_datetime_aware = datetime.now(user_tz)
    else:
        if reference_datetime.tzinfo is None or reference_datetime.tzinfo.utcoffset(reference_datetime) is None:
            reference_datetime_aware = user_tz.localize(reference_datetime)
        else:
            reference_datetime_aware = reference_datetime.astimezone(user_tz)
    
    reference_datetime_iso_str = reference_datetime_aware.isoformat()

    print(f"  Normalizing with LLM: '{natural_date_time_str}' (Reference: {reference_datetime_iso_str})")
    # normalize_datetime_with_llm is imported from tools.nlp_tools
    iso_datetime_str = normalize_datetime_with_llm(
        natural_date_time_str, 
        reference_datetime_iso_str, 
        user_timezone_str
    )

    if not iso_datetime_str:
        print(f"  LLM normalization failed for: '{natural_date_time_str}'")
        return None

    try:
        parsed_dt = dateutil_parser.parse(iso_datetime_str)
        if parsed_dt.tzinfo is None or parsed_dt.tzinfo.utcoffset(parsed_dt) is None:
            parsed_dt = user_tz.localize(parsed_dt)
        else:
            parsed_dt = parsed_dt.astimezone(user_tz)
        print(f"  Successfully parsed LLM output '{iso_datetime_str}' to: {parsed_dt.isoformat()}")
        return parsed_dt
    except Exception as e:
        print(f"  Could not parse LLM-normalized ISO string '{iso_datetime_str}': {e}")
        return None


def main():
    load_dotenv()
    config = load_config()

    user_timezone_str = config.get("timezone", "UTC") 
    print(f"Using timezone: {user_timezone_str}")
    try:
        user_tz = pytz.timezone(user_timezone_str)
    except pytz.exceptions.UnknownTimeZoneError:
        print(f"Error: Unknown timezone '{user_timezone_str}' in config.yaml. Defaulting to UTC.")
        user_timezone_str = "UTC"
        user_tz = pytz.utc
        
    current_time_user_tz = datetime.now(user_tz)
    
    agent_email_address = config.get("agent_email_address", "samirfawaz74@gmail.com")
    if agent_email_address == "samirfawaz74@gmail.com":
        print("CRITICAL: 'agent_email_address' not set in config.yaml. Please set it to your agent's primary Gmail address.")
        # return # Or use a default like samirfawaz74@gmail.com for testing if you ensure it's correct
        agent_email_address = "samirfawaz74@gmail.com" # Fallback for this example, user should configure
        print(f"Warning: Using fallback agent_email_address: {agent_email_address}")


    print("\n--- Step 1: Reading Recent Email ---")
    try:
        email_list_result = email_reader_tool.invoke("1") 

        if isinstance(email_list_result, str): 
            print(f"Email Reader Tool Error: {email_list_result}")
            return
        
        if not email_list_result:
            print("No emails found by the Email Reader Tool.")
            return

        latest_email = email_list_result[0]
        email_subject = latest_email.get("subject", "No Subject")
        email_body = latest_email.get("body_text", "No Body")
        
        from_details = latest_email.get("from_details", {})
        raw_email_sender_address = from_details.get("email") 
        if not raw_email_sender_address and latest_email.get("from"): # Fallback for old 'from' string
             # Basic parsing if 'from' is like "Name <email@addr>"
            match = re.search(r'<([^>]+)>', latest_email.get("from", ""))
            if match:
                raw_email_sender_address = match.group(1)
            elif "@" in latest_email.get("from", ""):
                 raw_email_sender_address = latest_email.get("from", "").strip()


        cc_recipients_data = latest_email.get("cc_recipients", []) 
        cc_email_list_for_analyzer = [item['email'] for item in cc_recipients_data if item.get('email')]

        print(f"Successfully read 1 email.")
        print(f"  From: {from_details.get('name', '')} <{raw_email_sender_address or 'Unknown'}>")
        if cc_email_list_for_analyzer:
            print(f"  Cc: {', '.join(cc_email_list_for_analyzer)}")
        print(f"  Subject: {email_subject}")
        print(f"  Body (snippet): {email_body[:150]}...")

        if not email_body.strip() and not email_subject.strip():
            print("Email has no subject or body content to analyze. Exiting.")
            return

        print("\n--- Step 2: Analyzing Email Content with Gemini ---")
        analyzer_input = {
            "email_subject": email_subject,
            "email_body": email_body,
            "cc_recipient_emails": cc_email_list_for_analyzer 
        }
        analysis_result = email_analyzer_tool.invoke(analyzer_input)

        print("\nAnalysis Result (JSON):")
        if isinstance(analysis_result, dict):
            print(json.dumps(analysis_result, indent=2))
            if analysis_result.get("error"):
                print(f"Error from analyzer tool: {analysis_result.get('error')}")
                print(f"Raw response was: {analysis_result.get('raw_response')}")
                return
        else:
            print(f"Analysis did not return a dictionary: {analysis_result}")
            return

        intent = analysis_result.get("intent")
        # Broadened intent check based on earlier testing
        if intent in ["schedule_new_meeting", "reschedule_meeting", "propose_new_time", "confirm_attendance"]:
            print(f"\n--- Step 3: Intent '{intent}' detected. Attempting to schedule/reschedule. ---")

            topic = analysis_result.get("topic") or email_subject 
            proposed_dates_times_str_list = analysis_result.get("proposed_dates_times")
            
            duration_from_analysis = analysis_result.get("duration_minutes")
            if isinstance(duration_from_analysis, int) and duration_from_analysis > 0:
                duration_minutes = duration_from_analysis
            else:
                duration_minutes = config.get("preferred_meeting_durations", [30])[0]
            
            # Attendee Logic
            meeting_attendees = []
            extracted_llm_attendees = analysis_result.get("attendees") # LLM now handles CC logic based on prompt
            if isinstance(extracted_llm_attendees, list):
                meeting_attendees.extend(attendee_email for attendee_email in extracted_llm_attendees if attendee_email and isinstance(attendee_email, str))

            # Add the original email sender if they are not the agent and not already in the list
            if raw_email_sender_address and raw_email_sender_address.lower() != agent_email_address.lower():
                if raw_email_sender_address not in meeting_attendees:
                    meeting_attendees.append(raw_email_sender_address)
                    print(f"  Added original email sender to attendees: {raw_email_sender_address}")
            
            meeting_attendees = sorted(list(set(meeting_attendees))) # Remove duplicates and sort for consistency
            print(f"  Final list of attendees for the calendar event: {meeting_attendees}")
            # --- End of Attendee Logic ---

            constraints_prefs = analysis_result.get("constraints_preferences")
            location_from_prefs = constraints_prefs if constraints_prefs is not None else "" 
            event_description = f"Meeting based on email from {from_details.get('name', '')} <{raw_email_sender_address or 'Unknown'}>.\n\nEmail Subject: {email_subject}\n\nExtracted Topic: {topic}\n\nConstraints/Preferences: {constraints_prefs or 'None'}"
            final_location = "Google Meet / Virtual" 
            if location_from_prefs: 
                if "video call" not in location_from_prefs.lower() and \
                   "virtual" not in location_from_prefs.lower() and \
                   location_from_prefs.strip():
                    final_location = location_from_prefs
            
            if not topic:
                print("No meeting topic identified. Cannot schedule.")
                return

            if not proposed_dates_times_str_list or not isinstance(proposed_dates_times_str_list, list) or not proposed_dates_times_str_list[0]:
                print("No specific proposed date/time found in the email to schedule directly, or format is incorrect.")
                return

            first_proposal_str = proposed_dates_times_str_list[0]
            print(f"Attempting to parse proposed date/time: '{first_proposal_str}'")
            
            start_datetime_obj = parse_datetime_from_llm(first_proposal_str, user_timezone_str, current_time_user_tz)

            if start_datetime_obj:
                end_datetime_obj = start_datetime_obj + timedelta(minutes=int(duration_minutes))
                start_iso = start_datetime_obj.isoformat()
                end_iso = end_datetime_obj.isoformat()

                print(f"Parsed Start: {start_iso}, End: {end_iso}, Timezone: {user_timezone_str}")

                if not meeting_attendees:
                    print("\nWarning: No attendees identified to invite. Scheduling event on your calendar only.")
                    # Optionally, add the agent itself as an attendee if no one else,
                    # or ensure there's at least one attendee (e.g. the agent)
                    # For now, we'll proceed, but Google Calendar might require at least the organizer.
                    # The organizer is implicit, so an empty attendee list for the API is okay.

                calendar_tool_input = {
                    "summary": topic,
                    "start_datetime_iso": start_iso,
                    "end_datetime_iso": end_iso,
                    "attendees": meeting_attendees, 
                    "description": event_description,
                    "location": final_location,
                    "timezone": user_timezone_str
                }
                                
                print(f"\nAttempting to create calendar event: '{topic}' for {start_iso}")
                creation_result = calendar_event_creator_tool.invoke(calendar_tool_input)
                print(f"Calendar Event Creation Result: {creation_result}")
            else:
                print(f"Could not parse any proposed date/time ('{first_proposal_str}') to schedule the meeting.")
        
        elif intent and intent != "not_meeting_related":
            print(f"\nEmail intent detected as '{intent}', but no direct scheduling action implemented for this in the test script.")
        else:
            print("\nEmail does not seem to be a meeting scheduling request based on analysis.")

    except Exception as e:
        print(f"An unexpected error occurred in the main workflow: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()