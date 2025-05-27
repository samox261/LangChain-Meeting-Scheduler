import os
import json
import yaml
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pytz
from dateutil import parser as dateutil_parser
from typing import Optional, Dict, Any, List
import re

# Import your LangChain tools
from tools.email_tools import email_reader_tool
from tools.nlp_tools import email_analyzer_tool, normalize_datetime_with_llm
from tools.calendar_tools import calendar_event_creator_tool

CONFIG_FILE = "config.yaml"
SCHEDULING_STATES_FILE = "scheduling_states.json"

# --- Configuration Loading ---
def load_config() -> Dict[str, Any]:
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
        if not config:
            print(f"Warning: {CONFIG_FILE} is empty.")
            return {}
        return config
    except FileNotFoundError:
        print(f"Warning: {CONFIG_FILE} not found.")
        return {}
    except Exception as e:
        print(f"Error loading {CONFIG_FILE}: {e}")
        return {}

# --- State Management Functions ---
def load_scheduling_states() -> Dict[str, Any]:
    if os.path.exists(SCHEDULING_STATES_FILE):
        try:
            with open(SCHEDULING_STATES_FILE, 'r') as f:
                states = json.load(f)
                print(f"Loaded {len(states)} scheduling states from {SCHEDULING_STATES_FILE}")
                return states
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading state file {SCHEDULING_STATES_FILE}: {e}. Starting with empty state.")
            return {}
    print(f"No state file found at {SCHEDULING_STATES_FILE}. Starting with empty state.")
    return {}

def save_scheduling_states(states_data: Dict[str, Any]):
    try:
        with open(SCHEDULING_STATES_FILE, 'w') as f:
            json.dump(states_data, f, indent=2)
        print(f"Saved {len(states_data)} scheduling states to {SCHEDULING_STATES_FILE}")
    except IOError as e:
        print(f"Error saving state file {SCHEDULING_STATES_FILE}: {e}")

# --- Helper Functions (is_valid_email, parse_datetime_from_llm) ---
def is_valid_email(email_str: Optional[str]) -> bool:
    if not isinstance(email_str, str):
        return False
    if re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email_str):
        return True
    return False

def parse_datetime_from_llm(
    natural_date_time_str: str,
    user_timezone_str: str,
    reference_datetime: Optional[datetime] = None
) -> Optional[datetime]:
    if not natural_date_time_str: return None
    user_tz = pytz.timezone(user_timezone_str)
    if reference_datetime is None:
        reference_datetime_aware = datetime.now(user_tz)
    else:
        if reference_datetime.tzinfo is None or reference_datetime.tzinfo.utcoffset(reference_datetime) is None:
            reference_datetime_aware = user_tz.localize(reference_datetime)
        else:
            reference_datetime_aware = reference_datetime.astimezone(user_tz)
    reference_datetime_iso_str = reference_datetime_aware.isoformat()
    # print(f"  Normalizing with LLM: '{natural_date_time_str}' (Reference: {reference_datetime_iso_str})") # Can be verbose
    iso_datetime_str = normalize_datetime_with_llm(
        natural_date_time_str, reference_datetime_iso_str, user_timezone_str
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

# --- Main Processing Function for a Single Email ---
def process_email(email_data: Dict[str, Any], agent_email_address: str, user_timezone_str: str, current_time_user_tz: datetime, config: Dict[str, Any], scheduling_states: Dict[str, Any]):
    """
    Processes a single email: analyzes it, updates state, and takes scheduling actions.
    Modifies scheduling_states in place.
    """
    email_id = email_data.get("id")
    thread_id = email_data.get("threadId")
    email_subject = email_data.get("subject", "No Subject")
    email_body = email_data.get("body_text", "No Body")
    from_details = email_data.get("from_details", {})
    raw_email_sender_address = from_details.get("email")
    if not raw_email_sender_address and email_data.get("from"): # Fallback
        match = re.search(r'<([^>]+)>', email_data.get("from", ""))
        if match: raw_email_sender_address = match.group(1)
        elif "@" in email_data.get("from", ""): raw_email_sender_address = email_data.get("from", "").strip()
    
    cc_recipients_data = email_data.get("cc_recipients", [])
    cc_email_list_for_analyzer = [item['email'] for item in cc_recipients_data if item.get('email')]

    print(f"\nProcessing Email ID: {email_id} (Thread ID: {thread_id})")
    print(f"  From: {from_details.get('name', '')} <{raw_email_sender_address or 'Unknown'}>")
    if cc_email_list_for_analyzer: print(f"  Cc: {', '.join(cc_email_list_for_analyzer)}")
    print(f"  Subject: {email_subject}")
    # print(f"  Body (snippet): {email_body[:150]}...") # Can be verbose

    if not email_body.strip() and not email_subject.strip():
        print("  Email has no subject or body content to analyze. Skipping.")
        return

    conversation_context_for_llm = "This is the first email perceived in this scheduling attempt for this thread."
    current_thread_state_data = scheduling_states.get(thread_id)
    if current_thread_state_data:
        print(f"  This email is part of an ongoing scheduling dialogue (Thread ID: {thread_id}).")
        status = current_thread_state_data.get('status', 'unknown')
        history = current_thread_state_data.get('negotiation_history', [])
        history_summary_parts = []
        for turn in history[-3:]:
            actor = turn.get('actor', 'unknown')
            message_summary = turn.get('message', turn.get('action', ''))
            if turn.get('details'): message_summary += f" (Details: {turn.get('details')})"
            history_summary_parts.append(f"{actor}: {message_summary}")
        history_summary = " | ".join(history_summary_parts)
        conversation_context_for_llm = f"This is a follow-up. Current status of this scheduling task: {status}. Recent dialogue: {history_summary}".strip()
        if not history_summary: conversation_context_for_llm = f"This is a follow-up. Current status: {status}. No detailed history in state."
    else:
        print(f"  This email appears to be the start of a new scheduling dialogue (Thread ID: {thread_id}).")

    print("\n  --- Analyzing Email Content with Gemini ---")
    analyzer_input = {
        "email_subject": email_subject,
        "email_body": email_body,
        "cc_recipient_emails": cc_email_list_for_analyzer,
        "conversation_context": conversation_context_for_llm
    }
    print(f"    Passing to analyzer. Context: {conversation_context_for_llm if conversation_context_for_llm else 'None'}")
    analysis_result = email_analyzer_tool.invoke(analyzer_input)

    print("\n  Analysis Result (JSON):")
    if isinstance(analysis_result, dict):
        print(json.dumps(analysis_result, indent=2))
        if analysis_result.get("error") or analysis_result.get("pydantic_validation_error"):
            error_msg = analysis_result.get("error", analysis_result.get("pydantic_validation_error"))
            print(f"  Error from analyzer tool: {error_msg}")
            if analysis_result.get('raw_response'): print(f"  Raw response was: {analysis_result.get('raw_response')}")
            return
    else:
        print(f"  Analysis did not return a dictionary: {analysis_result}")
        return

    intent = analysis_result.get("intent")
    actor_for_history = "external_party"
    if raw_email_sender_address and raw_email_sender_address.lower() == agent_email_address.lower():
        actor_for_history = "agent_self_email"

    if not current_thread_state_data and intent and intent != "not_meeting_related":
        current_thread_state_data = {
            "status": "new_request_analyzed", "topic": analysis_result.get("topic") or email_subject,
            "participants": [agent_email_address], "intent_history": [intent],
            "negotiation_history": [{"actor": actor_for_history, "message": f"EmailID: {email_id}, Subject: {email_subject}, BodySnippet: {email_body[:100]}...", "analysis": analysis_result}],
            "last_updated": datetime.now().isoformat()
        }
        if raw_email_sender_address and raw_email_sender_address.lower() != agent_email_address.lower():
            if raw_email_sender_address not in current_thread_state_data["participants"]:
                 current_thread_state_data["participants"].append(raw_email_sender_address)
        scheduling_states[thread_id] = current_thread_state_data
        print(f"    Created new state for thread {thread_id}.")
    elif current_thread_state_data and intent:
        current_thread_state_data["status"] = "follow_up_analyzed"
        current_thread_state_data.setdefault("intent_history", []).append(intent)
        current_thread_state_data.setdefault("negotiation_history", []).append(
            {"actor": actor_for_history, "message": f"EmailID: {email_id}, Subject: {email_subject}, BodySnippet: {email_body[:100]}...", "analysis": analysis_result})
        current_thread_state_data["last_reply_details"] = analysis_result
        current_thread_state_data["last_updated"] = datetime.now().isoformat()
        scheduling_states[thread_id] = current_thread_state_data
        print(f"    Updated state for thread {thread_id}.")
    # Save states after each email is processed
    save_scheduling_states(scheduling_states)


    # --- Conditional Scheduling Logic ---
    # TODO: This logic needs to be much more sophisticated based on current_thread_state_data and intent
    # For example, if intent is "reschedule_meeting", we need to find the old eventId and update it.
    # If intent is "confirm_attendance" to an agent's proposal, schedule it.
    # If intent is "propose_new_time" from user, agent needs to check availability and respond.

    scheduling_trigger_intents = ["schedule_new_meeting", "reschedule_meeting", "propose_new_time", "confirm_attendance"]
    if intent in scheduling_trigger_intents:
        print(f"\n  --- Processing Scheduling Action for Intent '{intent}' ---")
        topic = analysis_result.get("topic") or (current_thread_state_data.get("topic") if current_thread_state_data else None) or email_subject
        duration_from_analysis = analysis_result.get("duration_minutes")
        if isinstance(duration_from_analysis, int) and duration_from_analysis > 0:
            duration_minutes = duration_from_analysis
        else:
            duration_minutes = config.get("preferred_meeting_durations", [30])[0]
        
        final_meeting_attendees = []
        base_participants = []
        if current_thread_state_data and current_thread_state_data.get("participants"):
             base_participants.extend(p_email for p_email in current_thread_state_data.get("participants") if p_email)
        else: 
            base_participants.append(agent_email_address)
            if raw_email_sender_address and raw_email_sender_address.lower() != agent_email_address.lower():
                base_participants.append(raw_email_sender_address)
        final_meeting_attendees.extend(base_participants)
        llm_extracted_attendees = analysis_result.get("attendees")
        if isinstance(llm_extracted_attendees, list):
            for att_email in llm_extracted_attendees:
                if att_email and isinstance(att_email, str) and att_email.lower() not in [p.lower() for p in final_meeting_attendees]:
                    final_meeting_attendees.append(att_email)
        final_meeting_attendees = sorted(list(set(p.lower() for p in final_meeting_attendees if p)))
        valid_format_attendees = [email for email in final_meeting_attendees if is_valid_email(email)]
        print(f"    Final list of attendees for calendar event (validated): {valid_format_attendees}")
        
        constraints_prefs = analysis_result.get("constraints_preferences")
        location_from_prefs = constraints_prefs if constraints_prefs is not None else ""
        event_description = f"Meeting based on email thread (ID: {thread_id}).\nEmail Subject: {email_subject}\nExtracted Topic: {topic}\nConstraints/Preferences: {constraints_prefs or 'None'}"
        final_location = "Google Meet / Virtual"
        if location_from_prefs:
            if "video call" not in location_from_prefs.lower() and "virtual" not in location_from_prefs.lower() and location_from_prefs.strip():
                final_location = location_from_prefs
        if not topic:
            print("    No meeting topic identified. Cannot schedule.")
            return # from process_email

        proposed_dates_times_str_list = analysis_result.get("proposed_dates_times")
        if not proposed_dates_times_str_list or not isinstance(proposed_dates_times_str_list, list) or not proposed_dates_times_str_list[0]:
            print("    No specific proposed date/time found in the current email to schedule directly.")
            if current_thread_state_data: # Log that we need to act based on this
                current_thread_state_data["status"] = "needs_agent_proposal_or_clarification"
                current_thread_state_data.setdefault("negotiation_history", []).append(
                    {"actor": "agent", "action": "noted_no_time_proposal", "details": "Will need to propose times or ask for clarification."})
                save_scheduling_states(scheduling_states)
            return # from process_email

        first_proposal_str = proposed_dates_times_str_list[0]
        print(f"    Attempting to parse proposed date/time: '{first_proposal_str}'")
        start_datetime_obj = parse_datetime_from_llm(first_proposal_str, user_timezone_str, current_time_user_tz)

        if start_datetime_obj:
            end_datetime_obj = start_datetime_obj + timedelta(minutes=int(duration_minutes))
            start_iso = start_datetime_obj.isoformat()
            end_iso = end_datetime_obj.isoformat()
            print(f"    Parsed Start: {start_iso}, End: {end_iso}, Timezone: {user_timezone_str}")
            if not valid_format_attendees: print("\n    Warning: No valid attendees for calendar event.")
            
            calendar_tool_input = {
                "summary": topic, "start_datetime_iso": start_iso, "end_datetime_iso": end_iso,
                "attendees": valid_format_attendees, "description": event_description,
                "location": final_location, "timezone": user_timezone_str
            }
            
            # TODO: Implement actual RESCHEDULE logic here.
            # If intent is "reschedule_meeting" or "propose_new_time" (for an existing meeting)
            # AND current_thread_state_data contains a "google_calendar_event_id":
            #   - Call an UpdateCalendarEventTool instead of CreateCalendarEventTool.
            # For now, it will still create a NEW event if it gets this far.
            print(f"\n    Attempting to create/update calendar event: '{topic}' for {start_iso}")
            creation_result_dict = calendar_event_creator_tool.invoke(calendar_tool_input)
            print(f"    Calendar Event Creation Result: {creation_result_dict.get('message', 'No message')}")

            if current_thread_state_data and creation_result_dict.get("status") == "success":
                current_thread_state_data["status"] = "scheduled" # This needs to be more nuanced for reschedules
                current_thread_state_data.setdefault("negotiation_history", []).append(
                    {"actor": "agent", "action": "scheduled_meeting", 
                     "details": creation_result_dict.get('message'), 
                     "scheduled_time_iso": start_iso,
                     "eventId": creation_result_dict.get('eventId')})
                current_thread_state_data["scheduled_event_details"] = {
                    **calendar_tool_input,
                    "eventId": creation_result_dict.get('eventId'),
                    "htmlLink": creation_result_dict.get('htmlLink')
                }
                current_thread_state_data["last_updated"] = datetime.now().isoformat()
                save_scheduling_states(scheduling_states) # Save state after successful scheduling
                print(f"      Updated state for thread {thread_id} to 'scheduled' with Event ID: {creation_result_dict.get('eventId')}.")
            elif creation_result_dict.get("status") == "error":
                print(f"      Failed to create/update calendar event: {creation_result_dict.get('message')}")
                if current_thread_state_data:
                    current_thread_state_data["status"] = "scheduling_failed"
                    current_thread_state_data.setdefault("negotiation_history", []).append(
                        {"actor": "agent", "action": "scheduling_failed", "details": creation_result_dict.get('message')})
                    save_scheduling_states(scheduling_states)
        else:
            print(f"    Could not parse proposed date/time ('{first_proposal_str}') to schedule the meeting.")
            if current_thread_state_data: # Log that we need to act based on this
                current_thread_state_data["status"] = "date_parsing_failed"
                current_thread_state_data.setdefault("negotiation_history", []).append(
                    {"actor": "agent", "action": "date_parsing_failed", "details": f"Could not parse: {first_proposal_str}"})
                save_scheduling_states(scheduling_states)
    
    elif intent and intent != "not_meeting_related":
        print(f"\n  Intent detected as '{intent}'. No direct scheduling action taken by this script iteration.")
        if current_thread_state_data:
            current_thread_state_data.setdefault("negotiation_history", []).append(
                {"actor": "agent", "action": "acknowledged_intent", "intent_received": intent, "details": "Further agent logic TBD."})
            current_thread_state_data["status"] = f"user_intent_{intent}_received"
            save_scheduling_states(scheduling_states)
    else:
        print("\n  Email does not seem to be a meeting scheduling request based on analysis.")


def main_loop():
    """
    Main loop for the agent to periodically check and process emails.
    """
    load_dotenv()
    config = load_config()
    scheduling_states = load_scheduling_states()

    user_timezone_str = config.get("timezone", "UTC")
    print(f"Agent started. Using timezone: {user_timezone_str}")
    try:
        user_tz = pytz.timezone(user_timezone_str)
    except pytz.exceptions.UnknownTimeZoneError:
        print(f"Error: Unknown timezone '{user_timezone_str}' in config.yaml. Defaulting to UTC.")
        user_timezone_str = "UTC"
        user_tz = pytz.utc
    
    agent_email_address = config.get("agent_email_address", None)
    if not agent_email_address:
        print("CRITICAL: 'agent_email_address' not set in config.yaml. Exiting.")
        return

    print("\n--- Checking for new emails ---")
    try:
        # Fetch a small batch of recent unread emails, e.g., 5
        # The tool's lambda already handles parsing "5" to an int for max_results
        email_list_result = email_reader_tool.invoke("5") 

        if isinstance(email_list_result, str):
            print(f"Email Reader Tool Error: {email_list_result}")
            return # Exit if there's an error reading emails
        
        if not email_list_result:
            print("No new unread emails found to process.")
            return

        print(f"Found {len(email_list_result)} email(s) to potentially process.")
        # Process emails, typically newest first if the API returns them that way
        # For true robustness, we'd filter out emails already processed based on message ID
        # stored in our state, but for now, this loop processes what the reader tool gives.
        for email_data in reversed(email_list_result): # Process oldest of the batch first
            current_time_user_tz = datetime.now(user_tz) # Update current time for each email processing
            process_email(email_data, agent_email_address, user_timezone_str, current_time_user_tz, config, scheduling_states)
            print("--- Finished processing one email ---")

    except Exception as e:
        print(f"An unexpected error occurred in the main_loop: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # For a single run test:
    # main() # This was the old main function that just processed one email

    # For a loop that could be run periodically (still simplified):
    main_loop()
    print("\n--- Agent run complete ---")