import os
import json
import yaml
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pytz
from dateutil import parser as dateutil_parser
from typing import Optional, Dict, Any, List
import re
import time
import logging
import traceback

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("agent_single_user_continuous.log"),
        logging.StreamHandler()
    ]
)

# Import globally defined LangChain tools
from tools.email_tools import email_reader_tool, mark_as_read_tool
from tools.nlp_tools import email_analyzer_tool, normalize_datetime_with_llm
from tools.calendar_tools import calendar_event_creator_tool

CONFIG_FILE = "config.yaml"

def load_config() -> Dict[str, Any]:
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
        if not config: logging.warning(f"{CONFIG_FILE} is empty."); return {}
        return config
    except FileNotFoundError: logging.warning(f"{CONFIG_FILE} not found."); return {}
    except Exception as e: logging.error(f"Error loading {CONFIG_FILE}: {e}"); return {}

def get_user_specific_state_file_path(agent_email: str) -> str:
    safe_email_name = agent_email.replace('@','_at_').replace('.','_dot_')
    return f"scheduling_states_{safe_email_name}.json"

def load_scheduling_states(state_file_path: str) -> Dict[str, Any]:
    if os.path.exists(state_file_path):
        try:
            with open(state_file_path, 'r') as f:
                states = json.load(f)
                logging.info(f"Loaded {len(states)} states from {state_file_path}")
                return states
        except (IOError, json.JSONDecodeError) as e:
            logging.error(f"Error loading state file {state_file_path}: {e}. Starting fresh.")
            return {}
    logging.info(f"No state file at {state_file_path}. Starting fresh.")
    return {}

def save_scheduling_states(state_file_path: str, states_data: Dict[str, Any]):
    try:
        with open(state_file_path, 'w') as f:
            json.dump(states_data, f, indent=2)
        logging.info(f"Saved {len(states_data)} states to {state_file_path}")
    except IOError as e:
        logging.error(f"Error saving state file {state_file_path}: {e}")

def is_valid_email(email_str: Optional[str]) -> bool:
    if not isinstance(email_str, str): return False
    if re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email_str): return True
    return False

def parse_datetime_from_llm( natural_date_time_str: str, user_timezone_str: str, reference_datetime: Optional[datetime] = None) -> Optional[datetime]:
    if not natural_date_time_str: return None
    try:
        user_tz = pytz.timezone(user_timezone_str)
    except pytz.exceptions.UnknownTimeZoneError:
        logging.error(f"Unknown timezone '{user_timezone_str}' in parse_datetime_from_llm. Using UTC as fallback.")
        user_tz = pytz.utc
        user_timezone_str = "UTC"

    if reference_datetime is None:
        reference_datetime_aware = datetime.now(user_tz)
    else:
        if reference_datetime.tzinfo is None or reference_datetime.tzinfo.utcoffset(reference_datetime) is None:
            reference_datetime_aware = user_tz.localize(reference_datetime)
        else:
            reference_datetime_aware = reference_datetime.astimezone(user_tz)
    reference_datetime_iso_str = reference_datetime_aware.isoformat()
    logging.debug(f"Normalizing with LLM: '{natural_date_time_str}' (Reference: {reference_datetime_iso_str})")
    iso_datetime_str = normalize_datetime_with_llm(
        natural_date_time_str, reference_datetime_iso_str, user_timezone_str
    )
    if not iso_datetime_str:
        logging.warning(f"LLM normalization failed for: '{natural_date_time_str}'")
        return None
    try:
        parsed_dt = dateutil_parser.parse(iso_datetime_str)
        if parsed_dt.tzinfo is None or parsed_dt.tzinfo.utcoffset(parsed_dt) is None:
            parsed_dt = user_tz.localize(parsed_dt)
        else:
            parsed_dt = parsed_dt.astimezone(user_tz)
        logging.info(f"Successfully parsed LLM output '{iso_datetime_str}' to: {parsed_dt.isoformat()}")
        return parsed_dt
    except Exception as e:
        logging.error(f"Could not parse LLM-normalized ISO string '{iso_datetime_str}': {e}")
        return None

def process_single_email_for_agent(
    email_data: Dict[str, Any],
    agent_email_address: str,
    user_timezone_str: str,
    current_time_for_processing: datetime,
    config_settings: Dict[str, Any],
    agent_scheduling_states: Dict[str, Any], # This dict is mutable and changes will persist
    state_file_to_save: str # Used only for logging in this func, actual save is in main loop
    ):

    email_id = email_data.get("id")
    thread_id = email_data.get("threadId")

    email_subject = email_data.get("subject", "No Subject")
    email_body = email_data.get("body_text", "No Body")
    from_details = email_data.get("from_details", {})
    raw_email_sender_address = from_details.get("email")
    if not raw_email_sender_address and email_data.get("from"):
        match = re.search(r'<([^>]+)>', email_data.get("from", ""));
        if match: raw_email_sender_address = match.group(1)
        elif "@" in email_data.get("from", ""): raw_email_sender_address = email_data.get("from", "").strip()
    cc_recipients_data = email_data.get("cc_recipients", [])
    cc_email_list_for_analyzer = [item['email'] for item in cc_recipients_data if item.get('email')]

    logging.info(f"Processing Email ID: {email_id} for agent {agent_email_address} (Thread ID: {thread_id})")

    if not email_body.strip() and not email_subject.strip():
        logging.info("  Email has no subject or body content. Skipping.")
        if email_id:
            # Even if skipped, mark as read by agent and add to processed list
            try:
                mark_as_read_tool.invoke({"message_id": email_id})
                logging.info(f"  Marked empty email ID {email_id} as read by agent.")
                # Add to list of agent-processed emails
                processed_ids_list = agent_scheduling_states.setdefault("processed_message_ids", [])
                if email_id not in processed_ids_list:
                    processed_ids_list.append(email_id)
                    logging.info(f"  Email ID {email_id} (empty) added to agent's processed list.")
            except Exception as e_mark_empty:
                logging.error(f"  Failed to mark empty email ID {email_id} as read: {e_mark_empty}")
        return

    try:
        conversation_context_for_llm = "This is the first email perceived in this scheduling attempt for this thread."
        current_thread_state_data = agent_scheduling_states.get(thread_id)
        if current_thread_state_data:
            status = current_thread_state_data.get('status', 'unknown')
            history = current_thread_state_data.get('negotiation_history', [])
            history_summary_parts = [];
            for turn in history[-3:]:
                actor = turn.get('actor', 'unknown'); message_summary = turn.get('message', turn.get('action', ''))
                if turn.get('details'): message_summary += f" (Details: {turn.get('details')})"
                history_summary_parts.append(f"{actor}: {message_summary}")
            history_summary = " | ".join(history_summary_parts)
            conversation_context_for_llm = f"This is a follow-up. Current status: {status}. Recent dialogue: {history_summary}".strip()
            if not history_summary: conversation_context_for_llm = f"This is a follow-up. Current status: {status}. No detailed history."

        logging.info("  Analyzing Email Content with Gemini...")
        analyzer_input = {
            "email_subject": email_subject, "email_body": email_body,
            "cc_recipient_emails": cc_email_list_for_analyzer,
            "conversation_context": conversation_context_for_llm,
            "user_timezone_str": user_timezone_str
        }
        logging.info(f"    Passing to analyzer. User Timezone: {user_timezone_str}. Context snippet: {conversation_context_for_llm[:100]}...")
        analysis_result = email_analyzer_tool.invoke(analyzer_input)

        logging.info("  Analysis Result (JSON follows)")
        if isinstance(analysis_result, dict):
            logging.info(json.dumps(analysis_result, indent=2))
            if analysis_result.get("error") or analysis_result.get("pydantic_validation_error"):
                logging.error(f"  Error in analysis result: {analysis_result.get('error') or analysis_result.get('pydantic_validation_error')}")
                # Still mark as processed by agent to avoid retrying on error
                if email_id:
                    processed_ids_list = agent_scheduling_states.setdefault("processed_message_ids", [])
                    if email_id not in processed_ids_list:
                        processed_ids_list.append(email_id)
                return #
        else:
            logging.error(f"  Analysis result was not a dictionary: {analysis_result}")
            if email_id: # Still mark as processed
                processed_ids_list = agent_scheduling_states.setdefault("processed_message_ids", [])
                if email_id not in processed_ids_list:
                    processed_ids_list.append(email_id)
            return

        intent = analysis_result.get("intent")
        actor_for_history = "external_party"
        if raw_email_sender_address and raw_email_sender_address.lower() == agent_email_address.lower(): actor_for_history = "agent_self_email"
        
        # Update negotiation history before scheduling attempt
        if not current_thread_state_data and intent and intent != "not_meeting_related":
            current_thread_state_data = {
                "status": "new_request_analyzed", "topic": analysis_result.get("topic") or email_subject,
                "participants": [agent_email_address], "intent_history": [intent],
                "negotiation_history": [{"actor": actor_for_history, "message": f"EmailID: {email_id}, Subject: {email_subject}, Snippet: {email_body[:100]}...", "analysis": analysis_result}],
                "last_updated": datetime.now().isoformat()}
            if raw_email_sender_address and raw_email_sender_address.lower() != agent_email_address.lower():
                if raw_email_sender_address not in current_thread_state_data["participants"]:
                     current_thread_state_data["participants"].append(raw_email_sender_address)
            agent_scheduling_states[thread_id] = current_thread_state_data
            logging.info(f"    Created new state for thread {thread_id}.")
        elif current_thread_state_data and intent: #
            current_thread_state_data["status"] = "follow_up_analyzed"
            current_thread_state_data.setdefault("intent_history", []).append(intent)
            current_thread_state_data.setdefault("negotiation_history", []).append(
                {"actor": actor_for_history, "message": f"EmailID: {email_id}, Subject: {email_subject}, Snippet: {email_body[:100]}...", "analysis": analysis_result})
            current_thread_state_data["last_reply_details"] = analysis_result
            current_thread_state_data["last_updated"] = datetime.now().isoformat()
            # agent_scheduling_states[thread_id] = current_thread_state_data # agent_scheduling_states is mutable, already updated
            logging.info(f"    Updated state for thread {thread_id}.")
        # Note: save_scheduling_states is called in the main loop after the batch.

        scheduling_trigger_intents = ["schedule_new_meeting", "reschedule_meeting", "propose_new_time", "confirm_attendance"]
        if intent in scheduling_trigger_intents:
            logging.info(f"  Processing Scheduling Action for Intent '{intent}'...")
            topic = analysis_result.get("topic") or (current_thread_state_data.get("topic") if current_thread_state_data else None) or email_subject
            duration_from_analysis = analysis_result.get("duration_minutes")
            if isinstance(duration_from_analysis, int) and duration_from_analysis > 0: duration_minutes = duration_from_analysis
            else: duration_minutes = config_settings.get("preferred_meeting_durations", [30])[0]

            final_meeting_attendees = []
            base_participants = current_thread_state_data.get("participants", []) if current_thread_state_data else [agent_email_address]
            if agent_email_address not in base_participants: base_participants.append(agent_email_address)
            if raw_email_sender_address and raw_email_sender_address.lower() != agent_email_address.lower() and raw_email_sender_address not in base_participants:
                base_participants.append(raw_email_sender_address)
            final_meeting_attendees.extend(base_participants)
            llm_extracted_attendees = analysis_result.get("attendees")
            if isinstance(llm_extracted_attendees, list):
                for att_email in llm_extracted_attendees:
                    if att_email and isinstance(att_email, str) and att_email.lower() not in [p.lower() for p in final_meeting_attendees]:
                        final_meeting_attendees.append(att_email)
            final_meeting_attendees = sorted(list(set(p.lower() for p in final_meeting_attendees if p)))
            valid_format_attendees = [email for email in final_meeting_attendees if is_valid_email(email)]
            logging.info(f"    Final list of attendees for calendar event (validated): {valid_format_attendees}")

            constraints_prefs = analysis_result.get("constraints_preferences")
            location_from_prefs = constraints_prefs if constraints_prefs is not None else ""
            event_description = f"Meeting for {agent_email_address} from thread {thread_id}.\nSubject: {email_subject}\nTopic: {topic}\nDetails: {constraints_prefs or 'N/A'}"
            final_location = "Google Meet / Virtual"
            if location_from_prefs and "video call" not in location_from_prefs.lower() and "virtual" not in location_from_prefs.lower() and location_from_prefs.strip():
                final_location = location_from_prefs
            if not topic: logging.warning("    No meeting topic identified. Cannot schedule."); return

            already_scheduled_this_proposal = False
            if current_thread_state_data and current_thread_state_data.get("status") == "scheduled":
                if intent != "reschedule_meeting" and intent != "propose_new_time": #
                    logging.info(f"  Thread {thread_id} status is 'scheduled'. Intent '{intent}' might not warrant new action."); already_scheduled_this_proposal = True
            if already_scheduled_this_proposal: logging.info(f"  Skipping calendar event creation."); # No return here, still need to mark as processed below

            if not already_scheduled_this_proposal: # Only attempt scheduling if not already handled
                proposed_dates_times_str_list = analysis_result.get("proposed_dates_times")
                if not proposed_dates_times_str_list or not isinstance(proposed_dates_times_str_list, list) or not proposed_dates_times_str_list[0]:
                    logging.info("    No specific proposed date/time found in current email.")
                else:
                    first_proposal_str = proposed_dates_times_str_list[0]
                    logging.info(f"    Attempting to parse proposed date/time: '{first_proposal_str}'")
                    start_datetime_obj = parse_datetime_from_llm(first_proposal_str, user_timezone_str, current_time_for_processing)

                    if start_datetime_obj:
                        end_datetime_obj = start_datetime_obj + timedelta(minutes=int(duration_minutes))
                        start_iso = start_datetime_obj.isoformat(); end_iso = end_datetime_obj.isoformat()
                        logging.info(f"    Parsed Start: {start_iso}, End: {end_iso}, Timezone: {user_timezone_str}")
                        if not valid_format_attendees: logging.warning("    Warning: No valid attendees for calendar event.")
                        calendar_tool_input = {
                            "summary": topic, "start_datetime_iso": start_iso, "end_datetime_iso": end_iso,
                            "attendees": valid_format_attendees, "description": event_description,
                            "location": final_location, "timezone": user_timezone_str}
                        logging.info(f"    Attempting to create/update calendar event: '{topic}' for {start_iso}")
                        creation_result_dict = calendar_event_creator_tool.invoke(calendar_tool_input)
                        logging.info(f"    Calendar Event Creation Result: {creation_result_dict.get('message', 'No message')}")
                        if current_thread_state_data and creation_result_dict.get("status") == "success":
                            current_thread_state_data["status"] = "scheduled"
                            current_thread_state_data.setdefault("negotiation_history", []).append(
                                {"actor": "agent", "action": "scheduled_meeting", "details": creation_result_dict.get('message'),
                                 "scheduled_time_iso": start_iso, "eventId": creation_result_dict.get('eventId')})
                            current_thread_state_data["scheduled_event_details"] = {**calendar_tool_input, "eventId": creation_result_dict.get('eventId'), "htmlLink": creation_result_dict.get('htmlLink')}
                            current_thread_state_data["last_updated"] = datetime.now().isoformat()
                            # save_scheduling_states(state_file_to_save, agent_scheduling_states) # Saved in main loop
                            logging.info(f"      Updated state for thread {thread_id} to 'scheduled' with Event ID: {creation_result_dict.get('eventId')}.")
                        elif creation_result_dict.get("status") == "error": logging.error(f"      Failed to create/update calendar event: {creation_result_dict.get('message')}")
                    else: logging.warning(f"    Could not parse proposed date/time ('{first_proposal_str}').")
        elif intent and intent != "not_meeting_related":
            logging.info(f"  Intent detected as '{intent}'. No direct scheduling action this cycle.")
            if current_thread_state_data: current_thread_state_data["status"] = f"user_intent_{intent}_received" # ; save_scheduling_states(state_file_to_save, agent_scheduling_states)
        else:
            logging.info(f"  Email does not seem to be a meeting scheduling request.")

        # MODIFIED: Mark as read by agent AND add to agent's processed list
        if email_id:
            try:
                logging.info(f"  Attempting to mark email ID {email_id} as read by agent...")
                mark_as_read_tool.invoke({"message_id": email_id})
                logging.info(f"  Mark as read successful for email ID {email_id}")

                # Add to list of agent-processed emails
                # Ensures this happens for any email that reaches this point
                processed_ids_list = agent_scheduling_states.setdefault("processed_message_ids", [])
                if email_id not in processed_ids_list:
                    processed_ids_list.append(email_id) # agent_scheduling_states is a mutable dict
                    logging.info(f"  Email ID {email_id} added to agent's processed list.")
            except Exception as e_mark_read:
                logging.error(f"  Failed to mark email ID {email_id} as read or add to processed list: {e_mark_read}", exc_info=True)
    except Exception as e_outer_process:
        logging.error(f"Outer error processing email {email_id or 'unknown'} for agent {agent_email_address}: {e_outer_process}", exc_info=True)
        # If an error occurs during processing, still add its ID to prevent retrying problematic emails.
        if email_id:
            try:
                processed_ids_list = agent_scheduling_states.setdefault("processed_message_ids", [])
                if email_id not in processed_ids_list:
                    processed_ids_list.append(email_id)
                    logging.info(f"  Email ID {email_id} (error case) added to agent's processed list to prevent retries.")
            except Exception as e_add_processed_error:
                 logging.error(f"  Failed to add email ID {email_id} to processed list during error handling: {e_add_processed_error}")


def main_loop_for_single_user_continuous():
    load_dotenv()
    config = load_config()

    agent_email_address = config.get("agent_email_address", None)
    if not agent_email_address:
        logging.critical("CRITICAL: 'agent_email_address' not set in config.yaml. Agent cannot run.")
        return

    user_timezone_str = config.get("timezone", "UTC")
    logging.info(f"Agent loop started for {agent_email_address}. Using timezone: {user_timezone_str}")
    try:
        user_tz = pytz.timezone(user_timezone_str)
    except pytz.exceptions.UnknownTimeZoneError:
        logging.error(f"Error: Unknown timezone '{user_timezone_str}'. Defaulting to UTC.")
        user_timezone_str = "UTC"; user_tz = pytz.utc

    specific_state_file = get_user_specific_state_file_path(agent_email_address)
    POLL_INTERVAL_SECONDS = config.get("poll_interval_seconds", 300) #
    logging.info(f"Polling interval set to {POLL_INTERVAL_SECONDS} seconds.")

    while True:
        logging.info(f"--- Starting new check cycle for {agent_email_address} at {datetime.now(user_tz).strftime('%Y-%m-%d %H:%M:%S %Z')} ---")
        user_scheduling_states = load_scheduling_states(specific_state_file)
        # MODIFIED: Use a set for efficient lookup of processed IDs
        agent_processed_email_ids = set(user_scheduling_states.setdefault("processed_message_ids", []))

        try:
            email_list_result = email_reader_tool.invoke("5") # Fetches up to 5 recent emails

            if isinstance(email_list_result, str):
                logging.error(f"Email Reader Tool Error for {agent_email_address}: {email_list_result}")
            elif not email_list_result:
                logging.info(f"No new emails found for {agent_email_address} in this cycle by reader.")
            else:
                logging.info(f"Reader tool fetched {len(email_list_result)} email(s). Filtering against agent's processed list...")

                emails_to_process_this_cycle = []
                for email_data_item in email_list_result:
                    email_id = email_data_item.get("id")
                    if email_id and email_id not in agent_processed_email_ids:
                        emails_to_process_this_cycle.append(email_data_item)
                        logging.info(f"  Email ID {email_id} is new to the agent. Queued for processing.")
                    elif email_id:
                        logging.debug(f"  Skipping email ID {email_id} as it's already in agent's processed list.")
                
                if not emails_to_process_this_cycle:
                    logging.info(f"No new (unprocessed by agent) emails found for {agent_email_address} this cycle.")
                else:
                    logging.info(f"Processing {len(emails_to_process_this_cycle)} new email(s) for {agent_email_address} this cycle.")
                    # Process oldest of this new batch first
                    for email_data_item in reversed(emails_to_process_this_cycle):
                        current_time_for_processing = datetime.now(user_tz)
                        try:
                            # Pass the mutable user_scheduling_states so process_single_email_for_agent can update "processed_message_ids"
                            process_single_email_for_agent(
                                email_data_item,
                                agent_email_address,
                                user_timezone_str,
                                current_time_for_processing,
                                config,
                                user_scheduling_states, # Pass the main dict
                                specific_state_file
                            )
                        except Exception as e_process_email_loop:
                            logging.error(f"Error during process_single_email_for_agent for email ID {email_data_item.get('id')}: {e_process_email_loop}", exc_info=True)
                        logging.info(f"--- Finished one email processing iteration for {agent_email_address} ---")

                # Save states once after processing the batch for this user
                # This will include any updates to "processed_message_ids"
                save_scheduling_states(specific_state_file, user_scheduling_states)

        except Exception as e_user_cycle:
            logging.error(f"An unexpected error occurred in the processing cycle for {agent_email_address}: {e_user_cycle}", exc_info=True)

        logging.info(f"--- Cycle complete for {agent_email_address}. Waiting for {POLL_INTERVAL_SECONDS} seconds... ---")
        time.sleep(POLL_INTERVAL_SECONDS)

if __name__ == '__main__':
    main_loop_for_single_user_continuous()