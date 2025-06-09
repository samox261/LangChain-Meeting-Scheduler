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
        logging.FileHandler("assistant_mode.log"), # Log file for this mode
        logging.StreamHandler()
    ]
)

# Import LangChain tools
from tools.email_tools import email_reader_tool, mark_as_read_tool
from tools.nlp_tools import normalize_datetime_with_llm, parse_assistant_command # Make sure parse_assistant_command is in nlp_tools.py
from tools.calendar_tools import calendar_event_creator_tool, calendar_event_updater_tool, calendar_event_deleter_tool

CONFIG_FILE = "config.yaml"

def load_config() -> Dict[str, Any]:
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
        if not config: logging.warning(f"{CONFIG_FILE} is empty."); return {}
        return config
    except FileNotFoundError: logging.warning(f"{CONFIG_FILE} not found."); return {}
    except Exception as e: logging.error(f"Error loading {CONFIG_FILE}: {e}"); return {}

def get_user_specific_state_file_path(agent_email: str, mode_suffix: str = "assistant_commands") -> str:
    safe_email_name = agent_email.replace('@','_at_').replace('.','_dot_')
    return f"scheduling_states_{safe_email_name}_{mode_suffix}.json" # Unique state file

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

def parse_datetime_from_llm( natural_date_time_str: str, user_timezone_str: str, reference_datetime_for_normalization: Optional[datetime] = None) -> Optional[datetime]:
    if not natural_date_time_str: return None
    try:
        user_tz = pytz.timezone(user_timezone_str)
    except pytz.exceptions.UnknownTimeZoneError:
        logging.error(f"Unknown timezone '{user_timezone_str}' in parse_datetime_from_llm. Using UTC as fallback.")
        user_tz = pytz.utc
        user_timezone_str = "UTC"

    if reference_datetime_for_normalization is None:
        effective_reference_dt_aware = datetime.now(user_tz)
    else:
        if reference_datetime_for_normalization.tzinfo is None or reference_datetime_for_normalization.tzinfo.utcoffset(reference_datetime_for_normalization) is None:
            effective_reference_dt_aware = user_tz.localize(reference_datetime_for_normalization)
        else:
            effective_reference_dt_aware = reference_datetime_for_normalization.astimezone(user_tz)
            
    reference_datetime_iso_str = effective_reference_dt_aware.isoformat()
    logging.debug(f"Normalizing '{natural_date_time_str}' with LLM. Reference for LLM: {reference_datetime_iso_str}, Target TZ: {user_timezone_str}")
    
    iso_datetime_str = normalize_datetime_with_llm(
        natural_date_time_str, reference_datetime_iso_str, user_timezone_str
    )
    
    if not iso_datetime_str:
        logging.warning(f"LLM normalization failed for: '{natural_date_time_str}' using reference {reference_datetime_iso_str}")
        return None
    try:
        parsed_dt_naive = dateutil_parser.parse(iso_datetime_str)
        parsed_dt_aware = user_tz.localize(parsed_dt_naive)
        logging.info(f"Successfully parsed LLM output '{iso_datetime_str}' to: {parsed_dt_aware.isoformat()} (normalized in {user_timezone_str})")
        return parsed_dt_aware
    except Exception as e:
        logging.error(f"Could not parse LLM-normalized ISO string '{iso_datetime_str}': {e}")
        return None

# This is the function with the corrected topic handling
def process_assistant_command_from_email(
    email_data: Dict[str, Any],
    agent_email_address: str,
    user_timezone_str: str,
    current_time_for_email_processing: datetime,
    config_settings: Dict[str, Any], 
    assistant_states: Dict[str, Any]
):
    # Create timezone object at the start of the function
    try:
        user_tz = pytz.timezone(user_timezone_str)
    except pytz.exceptions.UnknownTimeZoneError:
        logging.error(f"Unknown timezone '{user_timezone_str}' in process_assistant_command_from_email. Using UTC as fallback.")
        user_tz = pytz.utc
        user_timezone_str = "UTC"

    email_id = email_data.get("id")
    thread_id = email_data.get("threadId")
    email_body = email_data.get("body_text", "")
    email_subject = email_data.get("subject", "No Subject") # Defaults to "No Subject" if missing
    from_details = email_data.get("from_details", {})
    email_sender = from_details.get("email")
    if not email_sender:
        from_header_raw = email_data.get("from", "")
        if from_header_raw:
            match = re.search(r'<([^>]+)>', from_header_raw)
            if match: email_sender = match.group(1)
            elif "@" in from_header_raw: email_sender = from_header_raw.strip()
    if not email_sender: email_sender = "unknown_sender@example.com"

    logging.info(f"ASSISTANT MODE: Processing command from Email ID: {email_id}, Thread ID: {thread_id}, Sender: {email_sender}")

    current_datetime_str_for_llm = current_time_for_email_processing.strftime("%A, %B %d, %Y, %I:%M %p %Z (%z)")
    parsed_command_data = parse_assistant_command(
        email_text_segment=email_body, 
        user_timezone_str=user_timezone_str,
        current_datetime_for_llm=current_datetime_str_for_llm
    )

    command = parsed_command_data.get("command")
    params = parsed_command_data.get("parameters") if parsed_command_data.get("parameters") is not None else {}
    
    logging.info(f"ASSISTANT MODE: Parsed Command: {command}, Parameters: {params}")
    if parsed_command_data.get("error_message"):
        logging.error(f"ASSISTANT MODE: Error parsing command: {parsed_command_data['error_message']}")

    current_thread_task_state = assistant_states.setdefault(thread_id, {
        "history": [], 
        "processed_command_email_ids_in_thread": [],
        "last_scheduled_event": {}
    })
    current_thread_task_state["history"].append({
        "email_id": email_id, "sender": email_sender,
        "parsed_command_data": parsed_command_data,
        "timestamp": datetime.now().isoformat()
    })

    action_taken_successfully = False

    if command == "SCHEDULE_MEETING":
        logging.info("ASSISTANT MODE: Executing SCHEDULE_MEETING command.")
        
        # --- MODIFIED TOPIC HANDLING (from response #104) ---
        topic_from_params = params.get("topic")
        if topic_from_params and topic_from_params.strip(): 
            topic = topic_from_params
        elif email_subject and email_subject != "No Subject": 
            topic = email_subject
        else:
            topic = "Scheduled Meeting" # Guaranteed fallback
        # --- END MODIFIED TOPIC HANDLING ---
        
        time_desc = params.get("time_description")
        attendees_text = params.get("attendees_text")

        if not time_desc:
            logging.warning("ASSISTANT MODE: No time_description provided for SCHEDULE_MEETING.")
            current_thread_task_state["history"][-1]["action_result"] = "Schedule failed: Missing time description."
        else:
            start_dt_obj = parse_datetime_from_llm(time_desc, user_timezone_str, current_time_for_email_processing)
            if start_dt_obj:
                duration_minutes = config_settings.get("preferred_meeting_durations", [30])[0]
                end_dt_obj = start_dt_obj + timedelta(minutes=duration_minutes)
                start_iso = start_dt_obj.isoformat()
                end_iso = end_dt_obj.isoformat()

                final_attendees = {agent_email_address.lower(), email_sender.lower()}
                if attendees_text:
                    if "cc" in attendees_text.lower() or "cc'd" in attendees_text.lower():
                        # Add all CC recipients
                        cc_recipients = email_data.get("cc_recipients", [])
                        for recipient in cc_recipients:
                            if recipient.get("email") and is_valid_email(recipient["email"]):
                                final_attendees.add(recipient["email"].lower())
                    else:
                        # Handle other attendee specifications
                        potential_emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', attendees_text)
                        for email in potential_emails:
                            if is_valid_email(email): 
                                final_attendees.add(email.lower())
                
                valid_attendees_list = sorted(list(final_attendees))
                logging.info(f"ASSISTANT MODE: Preparing to create event: Topic='{topic}', Start='{start_iso}', Attendees='{valid_attendees_list}'")
                
                creator_input = {
                    "summary": f"{topic} ({start_dt_obj.astimezone(user_tz).strftime('%I:%M %p')})",  # Convert to local timezone before formatting
                    "start_datetime_iso": start_iso, 
                    "end_datetime_iso": end_iso,
                    "timezone": user_timezone_str, 
                    "attendees": valid_attendees_list,
                    "description": f"Meeting scheduled by AI assistant via email command.\n\nCommand context:\n---\n{email_body[:300]}\n---",
                    "location": "Google Meet / Virtual"
                }
                creation_result = calendar_event_creator_tool.invoke(creator_input)
                logging.info(f"ASSISTANT MODE: Calendar event creation result: {creation_result}")
                
                if isinstance(creation_result, dict) and creation_result.get("status") == "success":
                    action_taken_successfully = True; event_id_created = creation_result.get("eventId")
                    current_thread_task_state["last_scheduled_event"] = {
                        "eventId": event_id_created, "summary": topic, "start_datetime_iso": start_iso,
                        "end_datetime_iso": end_iso, 
                        "htmlLink": creation_result.get("htmlLink"), "attendees": valid_attendees_list
                    }
                    current_thread_task_state["history"][-1]["action_result"] = f"Event Scheduled: ID {event_id_created}"
                else:
                    current_thread_task_state["history"][-1]["action_result"] = "Event Scheduling Failed: " + str(creation_result.get("message") if isinstance(creation_result, dict) else creation_result)
            else:
                logging.warning(f"ASSISTANT MODE: Could not parse time_description '{time_desc}' for SCHEDULE_MEETING.")
                current_thread_task_state["history"][-1]["action_result"] = f"Time parsing failed for: {time_desc}"

    elif command == "RESCHEDULE_MEETING":
        logging.info("ASSISTANT MODE: Executing RESCHEDULE_MEETING command.")
        meeting_id_text = params.get("meeting_identifier_text")
        new_time_desc = params.get("new_time_description")

        if not new_time_desc:
            logging.warning("ASSISTANT MODE: Missing new_time_description for RESCHEDULE_MEETING.")
            current_thread_task_state["history"][-1]["action_result"] = "Reschedule failed: Missing new time description."
        else:
            event_to_update_id = None; original_event_details_for_update = None
            if current_thread_task_state.get("last_scheduled_event", {}).get("eventId"):
                event_to_update_id = current_thread_task_state["last_scheduled_event"]["eventId"]
                original_event_details_for_update = current_thread_task_state["last_scheduled_event"]
                logging.info(f"ASSISTANT MODE: Found previous event in thread state to reschedule: ID {event_to_update_id}")
            else:
                logging.warning("ASSISTANT MODE: No 'last_scheduled_event' found in current thread state to identify meeting for reschedule.")
                current_thread_task_state["history"][-1]["action_result"] = "Reschedule failed: Could not identify meeting in thread."

            if event_to_update_id and original_event_details_for_update:
                original_start_dt_for_ref = None
                if original_event_details_for_update.get("start_datetime_iso"):
                    try: original_start_dt_for_ref = dateutil_parser.isoparse(original_event_details_for_update["start_datetime_iso"])
                    except ValueError: logging.warning(f"Could not parse original start time for reschedule ref: {original_event_details_for_update['start_datetime_iso']}")
                
                ref_dt_for_parsing = original_start_dt_for_ref if original_start_dt_for_ref else current_time_for_email_processing
                new_start_dt_obj = parse_datetime_from_llm(new_time_desc, user_timezone_str, ref_dt_for_parsing)

                if new_start_dt_obj:
                    original_duration_minutes = None
                    if original_event_details_for_update.get("start_datetime_iso") and original_event_details_for_update.get("end_datetime_iso"):
                        try:
                            s_dt = dateutil_parser.isoparse(original_event_details_for_update["start_datetime_iso"])
                            e_dt = dateutil_parser.isoparse(original_event_details_for_update["end_datetime_iso"])
                            original_duration_minutes = int((e_dt - s_dt).total_seconds() / 60)
                        except: pass
                    
                    duration_minutes = original_duration_minutes if original_duration_minutes is not None else config_settings.get("preferred_meeting_durations", [30])[0]
                    new_end_dt_obj = new_start_dt_obj + timedelta(minutes=duration_minutes)
                    new_start_iso = new_start_dt_obj.isoformat(); new_end_iso = new_end_dt_obj.isoformat()

                    updater_input = {
                        "event_id": event_to_update_id,
                        "summary": original_event_details_for_update.get("summary", "Meeting"),
                        "start_datetime_iso": new_start_iso, "end_datetime_iso": new_end_iso,
                        "timezone": user_timezone_str,
                        "attendees": original_event_details_for_update.get("attendees"),
                        "description": original_event_details_for_update.get("description"),
                        "location": original_event_details_for_update.get("location")
                    }
                    logging.info(f"ASSISTANT MODE: Preparing to update event ID {event_to_update_id} to Start: {new_start_iso}")
                    update_result = calendar_event_updater_tool.invoke(updater_input)
                    logging.info(f"ASSISTANT MODE: Calendar event update result: {update_result}")

                    if isinstance(update_result, dict) and update_result.get("status") == "success":
                        action_taken_successfully = True
                        current_thread_task_state["last_scheduled_event"].update({
                            "start_datetime_iso": new_start_iso, "end_datetime_iso": new_end_iso,
                            "htmlLink": update_result.get("htmlLink")
                        })
                        current_thread_task_state["history"][-1]["action_result"] = f"Event Rescheduled: ID {event_to_update_id}"
                    else:
                        current_thread_task_state["history"][-1]["action_result"] = "Event Rescheduling Failed: " + str(update_result.get("message") if isinstance(update_result, dict) else update_result)
                else:
                    logging.warning(f"ASSISTANT MODE: Could not parse new_time_description '{new_time_desc}' for RESCHEDULE_MEETING.")
                    current_thread_task_state["history"][-1]["action_result"] = f"New time parsing failed for: {new_time_desc}"

    elif command == "DELETE_MEETING":
        logging.info("ASSISTANT MODE: Executing DELETE_MEETING command.")
        meeting_id_text = params.get("meeting_identifier_text")
        
        event_to_delete_id = None
        if current_thread_task_state.get("last_scheduled_event", {}).get("eventId"):
            event_to_delete_id = current_thread_task_state["last_scheduled_event"]["eventId"]
            logging.info(f"ASSISTANT MODE: Identified event in thread state to delete: ID {event_to_delete_id} based on text: '{meeting_id_text if meeting_id_text else 'last in thread'}'")
        else:
            logging.warning(f"ASSISTANT MODE: Could not identify meeting to delete based on text '{meeting_id_text}' or last event in thread.")
            current_thread_task_state["history"][-1]["action_result"] = f"Delete failed: Could not identify meeting for '{meeting_id_text if meeting_id_text else 'any event in thread'}'."

        if event_to_delete_id:
            delete_result = calendar_event_deleter_tool.invoke({"event_id": event_to_delete_id})
            logging.info(f"ASSISTANT MODE: Calendar event deletion result: {delete_result}")
            
            if isinstance(delete_result, dict) and delete_result.get("status") == "success":
                action_taken_successfully = True
                current_thread_task_state["last_scheduled_event"] = {} 
                current_thread_task_state["history"][-1]["action_result"] = f"Event Deleted: ID {event_to_delete_id}"
            else:
                current_thread_task_state["history"][-1]["action_result"] = "Event Deletion Failed: " + str(delete_result.get("message") if isinstance(delete_result, dict) else delete_result)
                
    elif command == "UNKNOWN_COMMAND" or not command:
        logging.warning(f"ASSISTANT MODE: Received an unknown, unparseable, or empty command.")
        current_thread_task_state["history"][-1]["action_result"] = "Unknown command or parsing error."
    
    if command and command != "UNKNOWN_COMMAND" and not action_taken_successfully:
        logging.warning(f"ASSISTANT MODE: Command '{command}' was parsed but action was not successfully completed.")
        if "action_result" not in current_thread_task_state["history"][-1]:
            current_thread_task_state["history"][-1]["action_result"] = f"Command '{command}' parsed but not successfully executed."

    if email_id:
        try:
            mark_as_read_tool.invoke({"message_id": email_id})
            logging.info(f"ASSISTANT MODE: Email ID {email_id} marked as read.")
            processed_command_ids_global = assistant_states.setdefault("processed_command_email_ids", [])
            if email_id not in processed_command_ids_global:
                processed_command_ids_global.append(email_id)
                logging.info(f"ASSISTANT MODE: Email ID {email_id} added to assistant's global processed command list.")
            if email_id not in current_thread_task_state["processed_command_email_ids_in_thread"]:
                 current_thread_task_state["processed_command_email_ids_in_thread"].append(email_id)
        except Exception as e_mark_read:
            logging.error(f"ASSISTANT MODE: Failed to mark email ID {email_id} as read or add to processed list: {e_mark_read}", exc_info=True)


def main_assistant_mode_loop():
    load_dotenv()
    initial_config = load_config() 

    agent_email_address = initial_config.get("agent_email_address", None)
    if not agent_email_address:
        logging.critical("CRITICAL: 'agent_email_address' not set in config.yaml. Assistant cannot run.")
        return

    user_timezone_str = initial_config.get("timezone", "UTC")
    logging.info(f"ASSISTANT MODE: Loop started for {agent_email_address}. Using timezone: {user_timezone_str}")
    try:
        user_tz = pytz.timezone(user_timezone_str) 
    except pytz.exceptions.UnknownTimeZoneError:
        logging.error(f"ASSISTANT MODE: Error: Unknown timezone '{user_timezone_str}'. Defaulting to UTC.")
        user_timezone_str = "UTC"; user_tz = pytz.utc 

    specific_state_file = get_user_specific_state_file_path(agent_email_address, mode_suffix="assistant_commands")
    POLL_INTERVAL_SECONDS = initial_config.get("poll_interval_seconds", 150)
    logging.info(f"ASSISTANT MODE: Polling interval set to {POLL_INTERVAL_SECONDS} seconds.")

    while True:
        current_config = load_config() 
        assistant_settings = current_config.get("assistant_mode_settings", {})
        authorized_senders_list = assistant_settings.get("authorized_command_senders", [])
        
        if not authorized_senders_list:
            if "authorized_command_senders" not in assistant_settings : 
                 logging.warning("ASSISTANT MODE: 'authorized_command_senders' key missing in config.yaml under 'assistant_mode_settings'.")
            elif not authorized_senders_list: 
                 logging.warning("ASSISTANT MODE: 'authorized_command_senders' list is empty in config.yaml. Assistant might not respond to any commands this cycle.")

        authorized_senders = set(s.lower() for s in authorized_senders_list if isinstance(s, str) and s.strip())
        
        logging.info(f"--- ASSISTANT MODE: Starting new check cycle for {agent_email_address} at {datetime.now(user_tz).strftime('%Y-%m-%d %H:%M:%S %Z')} ---")
        assistant_states = load_scheduling_states(specific_state_file)
        agent_processed_command_email_ids = set(assistant_states.setdefault("processed_command_email_ids", []))
        current_time_for_this_cycle = datetime.now(user_tz) 

        try:
            email_list_result = email_reader_tool.invoke("5")

            if isinstance(email_list_result, str):
                logging.error(f"ASSISTANT MODE: Email Reader Tool Error for {agent_email_address}: {email_list_result}")
            elif not email_list_result:
                logging.info(f"ASSISTANT MODE: No new emails found for {agent_email_address} in this cycle by reader.")
            else:
                logging.info(f"ASSISTANT MODE: Reader tool fetched {len(email_list_result)} email(s).")
                emails_to_action_this_cycle = []

                for email_data_item in email_list_result:
                    email_id = email_data_item.get("id")
                    if email_id and email_id in agent_processed_command_email_ids:
                        logging.debug(f"  ASSISTANT MODE: Skipping email ID {email_id} as it's already globally processed.")
                        continue

                    from_details = email_data_item.get("from_details", {})
                    email_sender_address = from_details.get("email")
                    if not email_sender_address:
                        from_header_raw = email_data_item.get("from", "")
                        if from_header_raw:
                            match = re.search(r'<([^>]+)>', from_header_raw)
                            if match: email_sender_address = match.group(1)
                            elif "@" in from_header_raw: email_sender_address = from_header_raw.strip()
                    
                    agent_is_mentioned = False
                    email_body_for_check = email_data_item.get("body_text", "").lower()
                    cc_recipients = email_data_item.get("cc_recipients", [])

                    if agent_email_address.lower() in email_body_for_check: agent_is_mentioned = True
                    if not agent_is_mentioned:
                        for recipient in cc_recipients:
                            if recipient.get("email", "").lower() == agent_email_address.lower():
                                agent_is_mentioned = True; break
                    
                    if agent_is_mentioned:
                        logging.info(f"  ASSISTANT MODE: Agent mentioned in email ID {email_id} from sender '{email_sender_address}'.")
                        if email_sender_address and email_sender_address.lower() in authorized_senders:
                            logging.info(f"    Sender '{email_sender_address}' is AUTHORIZED. Queuing for command processing.")
                            emails_to_action_this_cycle.append(email_data_item)
                        else:
                            logging.warning(f"    Sender '{email_sender_address}' is NOT AUTHORIZED to command the assistant. Ignoring command in email ID {email_id}.")
                            if email_id:
                                global_processed_list = assistant_states.setdefault("processed_command_email_ids", [])
                                if email_id not in global_processed_list:
                                    global_processed_list.append(email_id)
                                logging.info(f"    Email ID {email_id} from unauthorized sender marked as processed to prevent re-evaluation.")
                    else:
                        logging.debug(f"  ASSISTANT MODE: Email ID {email_id} does not mention agent. Skipping.")

                if not emails_to_action_this_cycle:
                    logging.info(f"ASSISTANT MODE: No new, authorized emails with agent mention found for {agent_email_address} this cycle.")
                else:
                    logging.info(f"ASSISTANT MODE: Processing {len(emails_to_action_this_cycle)} authorized command email(s) for {agent_email_address} this cycle.")
                    for email_data_item in reversed(emails_to_action_this_cycle):
                        try:
                            process_assistant_command_from_email(
                                email_data_item, agent_email_address, user_timezone_str, 
                                current_time_for_this_cycle, current_config, assistant_states 
                            )
                        except Exception as e_process_command:
                            logging.error(f"ASSISTANT MODE: Error during process_assistant_command_from_email for email ID {email_data_item.get('id')}: {e_process_command}", exc_info=True)
                        logging.info(f"--- ASSISTANT MODE: Finished one command email processing iteration for {agent_email_address} ---")
                
                save_scheduling_states(specific_state_file, assistant_states)

        except Exception as e_user_cycle:
            logging.error(f"ASSISTANT MODE: An unexpected error in main loop: {e_user_cycle}", exc_info=True)

        logging.info(f"--- ASSISTANT MODE: Cycle complete for {agent_email_address}. Waiting for {POLL_INTERVAL_SECONDS} seconds... ---")
        time.sleep(POLL_INTERVAL_SECONDS)

if __name__ == '__main__':
    main_assistant_mode_loop()