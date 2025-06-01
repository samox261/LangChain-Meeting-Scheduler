import os
import json
import yaml
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pytz
from dateutil import parser as dateutil_parser # Ensure this is imported
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
from tools.calendar_tools import calendar_event_creator_tool, calendar_event_deleter_tool, calendar_event_updater_tool # MODIFIED

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

def parse_datetime_from_llm( natural_date_time_str: str, user_timezone_str: str, reference_datetime_for_normalization: Optional[datetime] = None) -> Optional[datetime]: # Renamed reference_datetime for clarity
    if not natural_date_time_str: return None
    try:
        user_tz = pytz.timezone(user_timezone_str)
    except pytz.exceptions.UnknownTimeZoneError:
        logging.error(f"Unknown timezone '{user_timezone_str}' in parse_datetime_from_llm. Using UTC as fallback.")
        user_tz = pytz.utc
        user_timezone_str = "UTC"

    # Use the provided reference_datetime_for_normalization for the LLM, otherwise default to now.
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
        natural_date_time_str, reference_datetime_iso_str, user_timezone_str # Pass user_timezone_str as target_timezone_str
    )
    
    if not iso_datetime_str:
        logging.warning(f"LLM normalization failed for: '{natural_date_time_str}' using reference {reference_datetime_iso_str}")
        return None
    try:
        # The LLM should return YYYY-MM-DDTHH:MM:SS in the target_timezone_str (but without offset in the string)
        # dateutil.parser can parse this. We then localize it to ensure it's an aware datetime object.
        parsed_dt_naive = dateutil_parser.parse(iso_datetime_str)
        parsed_dt_aware = user_tz.localize(parsed_dt_naive) # Localize to the target timezone
        logging.info(f"Successfully parsed LLM output '{iso_datetime_str}' to: {parsed_dt_aware.isoformat()} (normalized in {user_timezone_str})")
        return parsed_dt_aware
    except Exception as e:
        logging.error(f"Could not parse LLM-normalized ISO string '{iso_datetime_str}': {e}")
        return None

def process_single_email_for_agent(
    email_data: Dict[str, Any],
    agent_email_address: str,
    user_timezone_str: str,
    current_time_for_processing: datetime, # This is "now" when the email is processed
    config_settings: Dict[str, Any],
    agent_scheduling_states: Dict[str, Any],
    state_file_to_save: str
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
            try:
                mark_as_read_tool.invoke({"message_id": email_id})
                logging.info(f"  Marked empty email ID {email_id} as read by agent.")
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
        original_event_start_for_reference = None # For date parsing context

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
            
            # Get original event start time if available, for date parsing reference in reschedules
            if current_thread_state_data.get("scheduled_event_details", {}).get("start_datetime_iso"):
                try:
                    original_event_start_for_reference = dateutil_parser.isoparse(current_thread_state_data["scheduled_event_details"]["start_datetime_iso"])
                except ValueError:
                    logging.warning(f"Could not parse original event start ISO: {current_thread_state_data['scheduled_event_details']['start_datetime_iso']}")


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
                if email_id:
                    processed_ids_list = agent_scheduling_states.setdefault("processed_message_ids", [])
                    if email_id not in processed_ids_list:
                        processed_ids_list.append(email_id)
                return
        else:
            logging.error(f"  Analysis result was not a dictionary: {analysis_result}")
            if email_id:
                processed_ids_list = agent_scheduling_states.setdefault("processed_message_ids", [])
                if email_id not in processed_ids_list:
                    processed_ids_list.append(email_id)
            return

        intent = analysis_result.get("intent")
        actor_for_history = "external_party"
        if raw_email_sender_address and raw_email_sender_address.lower() == agent_email_address.lower(): actor_for_history = "agent_self_email"
        
        if not current_thread_state_data and intent and intent != "not_meeting_related":
            current_thread_state_data = {
                "status": "new_request_analyzed", 
                "topic": analysis_result.get("topic") or email_subject,
                "participants": [agent_email_address], 
                "intent_history": [intent],
                "negotiation_history": [{"actor": actor_for_history, "message": f"EmailID: {email_id}, Subject: {email_subject}, Snippet: {email_body[:100]}...", "analysis": analysis_result, "timestamp": datetime.now().isoformat()}],
                "last_updated": datetime.now().isoformat()}
            if raw_email_sender_address and raw_email_sender_address.lower() != agent_email_address.lower():
                if raw_email_sender_address not in current_thread_state_data["participants"]:
                     current_thread_state_data["participants"].append(raw_email_sender_address)
            agent_scheduling_states[thread_id] = current_thread_state_data
            logging.info(f"    Created new state for thread {thread_id}.")
        elif current_thread_state_data and intent:
            # Ensure last_reply_details is updated before potential early returns
            current_thread_state_data["last_reply_details"] = analysis_result 
            # Don't change status yet, let the scheduling logic do that
            current_thread_state_data.setdefault("intent_history", []).append(intent)
            current_thread_state_data.setdefault("negotiation_history", []).append(
                {"actor": actor_for_history, "message": f"EmailID: {email_id}, Subject: {email_subject}, Snippet: {email_body[:100]}...", "analysis": analysis_result, "timestamp": datetime.now().isoformat()})
            current_thread_state_data["last_updated"] = datetime.now().isoformat()
            logging.info(f"    Updated state for thread {thread_id} with new analysis.")

        # --- SCHEDULING LOGIC ---
        scheduling_trigger_intents = ["schedule_new_meeting", "reschedule_meeting", "propose_new_time", "confirm_attendance"]
        if intent in scheduling_trigger_intents:
            logging.info(f"  Processing Calendar Action for Intent '{intent}'...")

            # Prepare event details from analysis and current state
            # Topic: Use LLM analysis, fallback to current state, then subject
            current_topic = analysis_result.get("topic") 
            if not current_topic and current_thread_state_data:
                current_topic = current_thread_state_data.get("topic")
            if not current_topic:
                current_topic = email_subject
            if not current_topic: # Still no topic
                logging.warning("    No meeting topic identified. Cannot schedule or update.")
                if email_id: # Mark as processed even if we can't schedule
                    processed_ids_list = agent_scheduling_states.setdefault("processed_message_ids", [])
                    if email_id not in processed_ids_list: processed_ids_list.append(email_id)
                return
            
            # Duration: Use LLM analysis, fallback to config
            duration_from_analysis = analysis_result.get("duration_minutes")
            if isinstance(duration_from_analysis, int) and duration_from_analysis > 0: 
                current_duration_minutes = duration_from_analysis
            elif current_thread_state_data and current_thread_state_data.get("scheduled_event_details", {}).get("start_datetime_iso") and current_thread_state_data.get("scheduled_event_details", {}).get("end_datetime_iso"):
                # If not in analysis, try to derive from existing event if it's a reschedule
                try:
                    s_dt = dateutil_parser.isoparse(current_thread_state_data["scheduled_event_details"]["start_datetime_iso"])
                    e_dt = dateutil_parser.isoparse(current_thread_state_data["scheduled_event_details"]["end_datetime_iso"])
                    current_duration_minutes = int((e_dt - s_dt).total_seconds() / 60)
                except:
                    current_duration_minutes = config_settings.get("preferred_meeting_durations", [30])[0]
            else:
                current_duration_minutes = config_settings.get("preferred_meeting_durations", [30])[0]

            # Attendees: Merge from current state, LLM analysis, and sender/agent
            final_meeting_attendees = []
            base_participants = current_thread_state_data.get("participants", [agent_email_address]) if current_thread_state_data else [agent_email_address]
            if agent_email_address not in base_participants: base_participants.append(agent_email_address)
            if raw_email_sender_address and raw_email_sender_address.lower() != agent_email_address.lower() and raw_email_sender_address not in base_participants:
                base_participants.append(raw_email_sender_address)
            final_meeting_attendees.extend(base_participants)
            
            llm_extracted_attendees = analysis_result.get("attendees") # This is from the current email
            if isinstance(llm_extracted_attendees, list): # If LLM provides new list, it might be the definitive one for reschedule
                # For a reschedule, if LLM provides attendees, it should be the new set.
                # If it's a new meeting, these are additive.
                # Let's assume for now if LLM gives attendees, these are the ones to use for the event.
                # More sophisticated merging might be needed.
                for att_email in llm_extracted_attendees:
                    if att_email and isinstance(att_email, str) and att_email.lower() not in [p.lower() for p in final_meeting_attendees]:
                        final_meeting_attendees.append(att_email)
            elif current_thread_state_data and current_thread_state_data.get("scheduled_event_details", {}).get("attendees"):
                 # If LLM didn't specify attendees for a reschedule, keep the old ones
                for att_email in current_thread_state_data["scheduled_event_details"]["attendees"]:
                     if att_email and isinstance(att_email, str) and att_email.lower() not in [p.lower() for p in final_meeting_attendees]:
                        final_meeting_attendees.append(att_email)

            final_meeting_attendees = sorted(list(set(p.lower() for p in final_meeting_attendees if p and is_valid_email(p))))
            logging.info(f"    Final list of attendees for calendar event: {final_meeting_attendees}")

            # Description and Location
            current_description = analysis_result.get("description")
            if current_description is None and current_thread_state_data and current_thread_state_data.get("scheduled_event_details"): #
                current_description = current_thread_state_data.get("scheduled_event_details",{}).get("description")
            if current_description is None : # Default description
                current_description = f"Meeting regarding: {current_topic}\nFacilitated by scheduling agent."


            current_location = analysis_result.get("location")
            if current_location is None and current_thread_state_data and current_thread_state_data.get("scheduled_event_details"): #
                current_location = current_thread_state_data.get("scheduled_event_details",{}).get("location")
            if current_location is None: # Default location
                current_location = "Google Meet / Virtual"


            # --- RESCHEDULE (UPDATE) vs. CREATE NEW ---
            is_reschedule_or_update = intent in ["reschedule_meeting", "propose_new_time"] and \
                                    current_thread_state_data and \
                                    current_thread_state_data.get("status") == "scheduled" and \
                                    current_thread_state_data.get("scheduled_event_details", {}).get("eventId")

            if is_reschedule_or_update:
                existing_event_id = current_thread_state_data["scheduled_event_details"]["eventId"]
                logging.info(f"  Attempting to UPDATE existing event ID: {existing_event_id} for reschedule.")

                new_proposed_times_str_list = analysis_result.get("proposed_dates_times")
                if not new_proposed_times_str_list or not new_proposed_times_str_list[0]:
                    logging.warning("    Reschedule requested, but no new time found in analysis. Cannot update.")
                else:
                    new_time_proposal_str = new_proposed_times_str_list[0]
                    # Use original event's start as reference for parsing new time
                    reference_for_date_parsing = original_event_start_for_reference if original_event_start_for_reference else current_time_for_processing
                    
                    new_start_datetime_obj = parse_datetime_from_llm(new_time_proposal_str, user_timezone_str, reference_for_date_parsing)

                    if new_start_datetime_obj:
                        new_end_datetime_obj = new_start_datetime_obj + timedelta(minutes=int(current_duration_minutes))
                        new_start_iso = new_start_datetime_obj.isoformat()
                        new_end_iso = new_end_datetime_obj.isoformat()
                        logging.info(f"    Parsed new time for update: Start ISO: {new_start_iso}, End ISO: {new_end_iso}")

                        update_input = {
                            "event_id": existing_event_id,
                            "summary": current_topic, # Pass current topic
                            "start_datetime_iso": new_start_iso,
                            "end_datetime_iso": new_end_iso,
                            "timezone": user_timezone_str, # Pass user's timezone
                            "attendees": final_meeting_attendees, # Pass merged list
                            "description": current_description, # Pass current/updated description
                            "location": current_location # Pass current/updated location
                        }
                        update_result = calendar_event_updater_tool.invoke(update_input)
                        log_message_update = update_result.get('message', 'No message') if isinstance(update_result, dict) else str(update_result)
                        logging.info(f"    Update event {existing_event_id} result: {log_message_update}")

                        history_action = "updated_scheduled_meeting"
                        history_details = f"Original Event ID: {existing_event_id}. Result: {log_message_update}"
                        if isinstance(update_result, dict) and update_result.get("status") == "success":
                            current_thread_state_data["status"] = "scheduled" # Remains scheduled, but updated
                            current_thread_state_data["scheduled_event_details"] = {
                                "eventId": update_result.get("eventId", existing_event_id), # Should be same ID
                                "summary": current_topic,
                                "start_datetime_iso": new_start_iso,
                                "end_datetime_iso": new_end_iso,
                                "timezone": user_timezone_str,
                                "attendees": final_meeting_attendees,
                                "description": current_description,
                                "location": current_location,
                                "htmlLink": update_result.get("htmlLink")
                            }
                        else:
                            history_action = "update_scheduled_meeting_failed"
                            logging.error(f"    Failed to update event {existing_event_id}.")
                        
                        current_thread_state_data.setdefault("negotiation_history", []).append(
                            {"actor": "agent", "action": history_action, "details": history_details, "timestamp": datetime.now().isoformat()})
                    else:
                        logging.warning(f"    Could not parse new proposed time ('{new_time_proposal_str}') for reschedule.")
                        current_thread_state_data.setdefault("negotiation_history", []).append(
                            {"actor": "agent", "action": "datetime_parse_failed_for_update", "details": f"Could not parse: {new_time_proposal_str}", "timestamp": datetime.now().isoformat()})
            
            elif intent == "schedule_new_meeting" or (intent in ["propose_new_time", "reschedule_meeting"] and not is_reschedule_or_update) : # Create new event
                # This block handles new meeting requests, or propose_new_time/reschedule if no existing event was found to update.
                logging.info("  Attempting to CREATE a new calendar event.")
                proposed_dates_times_str_list = analysis_result.get("proposed_dates_times")
                if not proposed_dates_times_str_list or not isinstance(proposed_dates_times_str_list, list) or not proposed_dates_times_str_list[0]:
                    logging.info("    No specific proposed date/time found in current email for new event.")
                else:
                    first_proposal_str = proposed_dates_times_str_list[0]
                    # For a brand new meeting, reference time is current processing time
                    start_datetime_obj = parse_datetime_from_llm(first_proposal_str, user_timezone_str, current_time_for_processing)

                    if start_datetime_obj:
                        end_datetime_obj = start_datetime_obj + timedelta(minutes=int(current_duration_minutes))
                        start_iso = start_datetime_obj.isoformat(); end_iso = end_datetime_obj.isoformat()
                        logging.info(f"    Parsed Start for new event: {start_iso}, End: {end_iso}")
                        
                        calendar_tool_input = {
                            "summary": current_topic, "start_datetime_iso": start_iso, "end_datetime_iso": end_iso,
                            "attendees": final_meeting_attendees, "description": current_description,
                            "location": current_location, "timezone": user_timezone_str}
                        creation_result_dict = calendar_event_creator_tool.invoke(calendar_tool_input)
                        logging.info(f"    New Calendar Event Creation Result: {creation_result_dict.get('message', 'No message')}")
                        
                        if current_thread_state_data and creation_result_dict.get("status") == "success":
                            current_thread_state_data["status"] = "scheduled"
                            current_thread_state_data.setdefault("negotiation_history", []).append(
                                {"actor": "agent", "action": "scheduled_new_meeting", "details": creation_result_dict.get('message'), 
                                 "scheduled_time_iso": start_iso, "eventId": creation_result_dict.get('eventId'), "timestamp": datetime.now().isoformat()})
                            # Store all details used for creation for future reference/updates
                            current_thread_state_data["scheduled_event_details"] = {
                                "eventId": creation_result_dict.get('eventId'),
                                "summary": current_topic,
                                "start_datetime_iso": start_iso,
                                "end_datetime_iso": end_iso,
                                "timezone": user_timezone_str,
                                "attendees": final_meeting_attendees, # Store the actual list used
                                "description": current_description,
                                "location": current_location,
                                "htmlLink": creation_result_dict.get('htmlLink')
                            }
                            current_thread_state_data["last_updated"] = datetime.now().isoformat()
                        elif creation_result_dict.get("status") == "error":
                            logging.error(f"      Failed to create new calendar event: {creation_result_dict.get('message')}")
                            current_thread_state_data.setdefault("negotiation_history", []).append(
                                {"actor": "agent", "action": "schedule_new_meeting_failed", "details": creation_result_dict.get('message'), "timestamp": datetime.now().isoformat()})
                    else: 
                        logging.warning(f"    Could not parse proposed date/time ('{first_proposal_str}') for new event.")
                        current_thread_state_data.setdefault("negotiation_history", []).append(
                                {"actor": "agent", "action": "datetime_parse_failed_for_create", "details": f"Could not parse: {first_proposal_str}", "timestamp": datetime.now().isoformat()})
            
            elif intent == "confirm_attendance":
                logging.info("  Intent is 'confirm_attendance'. No calendar action taken by agent, assuming user/organizer handles this.")
                if current_thread_state_data: current_thread_state_data["status"] = "attendance_confirmed_by_other_party"
            
            # Other intents like "cancel_meeting", "decline_attendance" might need specific calendar actions (e.g., using deleter_tool)
            # For now, they just update history and state.

        elif intent and intent != "not_meeting_related": # Non-scheduling intents that are meeting related
            logging.info(f"  Intent detected as '{intent}'. No direct calendar modification this cycle.")
            if current_thread_state_data: current_thread_state_data["status"] = f"user_intent_{intent}_received"
        else: # Not meeting related
            logging.info(f"  Email does not seem to be a meeting scheduling request or directly related action.")

        # Mark as processed by agent (add to processed_message_ids list)
        if email_id:
            try:
                logging.info(f"  Attempting to mark email ID {email_id} as read by agent...")
                mark_as_read_tool.invoke({"message_id": email_id})
                logging.info(f"  Mark as read successful for email ID {email_id}")

                processed_ids_list = agent_scheduling_states.setdefault("processed_message_ids", [])
                if email_id not in processed_ids_list:
                    processed_ids_list.append(email_id)
                    logging.info(f"  Email ID {email_id} added to agent's processed list.")
            except Exception as e_mark_read:
                logging.error(f"  Failed to mark email ID {email_id} as read or add to processed list: {e_mark_read}", exc_info=True)
    except Exception as e_outer_process:
        logging.error(f"Outer error processing email {email_id or 'unknown'} for agent {agent_email_address}: {e_outer_process}", exc_info=True)
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
    POLL_INTERVAL_SECONDS = config.get("poll_interval_seconds", 150) 
    logging.info(f"Polling interval set to {POLL_INTERVAL_SECONDS} seconds.")

    while True:
        logging.info(f"--- Starting new check cycle for {agent_email_address} at {datetime.now(user_tz).strftime('%Y-%m-%d %H:%M:%S %Z')} ---")
        user_scheduling_states = load_scheduling_states(specific_state_file)
        agent_processed_email_ids = set(user_scheduling_states.setdefault("processed_message_ids", []))

        try:
            email_list_result = email_reader_tool.invoke("5")

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
                    for email_data_item in reversed(emails_to_process_this_cycle):
                        current_time_for_processing = datetime.now(user_tz)
                        try:
                            process_single_email_for_agent(
                                email_data_item,
                                agent_email_address,
                                user_timezone_str,
                                current_time_for_processing,
                                config,
                                user_scheduling_states, 
                                specific_state_file
                            )
                        except Exception as e_process_email_loop:
                            logging.error(f"Error during process_single_email_for_agent for email ID {email_data_item.get('id')}: {e_process_email_loop}", exc_info=True)
                        logging.info(f"--- Finished one email processing iteration for {agent_email_address} ---")
                save_scheduling_states(specific_state_file, user_scheduling_states)

        except Exception as e_user_cycle:
            logging.error(f"An unexpected error occurred in the processing cycle for {agent_email_address}: {e_user_cycle}", exc_info=True)

        logging.info(f"--- Cycle complete for {agent_email_address}. Waiting for {POLL_INTERVAL_SECONDS} seconds... ---")
        time.sleep(POLL_INTERVAL_SECONDS)

if __name__ == '__main__':
    main_loop_for_single_user_continuous()