import os.path
from typing import Optional, List, Dict, Any
import logging

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

# Import SCOPES from email_tools to ensure consistency for token.json
from tools.email_tools import SCOPES

CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.json' # Single token file

def get_calendar_service() -> Optional[build]:
    creds = None
    if os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
            if not creds or not creds.has_scopes(['https://www.googleapis.com/auth/calendar']):
                logging.warning(f"Token loaded from {TOKEN_FILE} but may be missing calendar scope. Re-check or re-auth if errors occur.")
        except Exception as e:
            logging.warning(f"Could not load token from {TOKEN_FILE} for calendar (may need re-auth or scopes changed): {e}")
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                logging.info(f"Refreshing token (calendar context)...")
                creds.refresh(Request())
                logging.info(f"Token refreshed successfully (calendar context).")
            except Exception as e:
                logging.warning(f"Error refreshing token (calendar context): {e}. Re-authenticating.")
                creds = None

        if not creds:
            logging.info(f"Initiating new OAuth flow (calendar context, using combined scopes: {SCOPES})...")
            if not os.path.exists(CREDENTIALS_FILE):
                logging.error(f"CRITICAL: {CREDENTIALS_FILE} not found. Cannot initiate OAuth flow.")
                return None
            try:
                flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
                creds = flow.run_local_server(port=0)
            except Exception as e_flow:
                logging.error(f"OAuth flow failed (calendar context): {e_flow}", exc_info=True)
                return None

        if creds:
            with open(TOKEN_FILE, 'w') as token_file_handle:
                token_file_handle.write(creds.to_json())
            logging.info(f"Credentials saved to {TOKEN_FILE} (calendar context).")
        else:
            logging.error(f"Could not obtain credentials after OAuth flow (calendar context).")
            return None

    try:
        service = build('calendar', 'v3', credentials=creds)
        logging.debug("Google Calendar service client built successfully.")
        return service
    except HttpError as error:
        logging.error(f'An HttpError occurred while building the calendar service: {error}')
        return None
    except Exception as e:
        logging.error(f'An unexpected error in get_calendar_service: {str(e)}', exc_info=True)
        return None

class CreateCalendarEventInput(BaseModel):
    summary: str = Field(description="The title or summary of the calendar event.")
    start_datetime_iso: str = Field(description="The start date and time in ISO 8601 format.")
    end_datetime_iso: str = Field(description="The end date and time in ISO 8601 format.")
    timezone: str = Field(description="The timezone for the event (e.g., 'Asia/Beirut').")
    attendees: Optional[List[str]] = Field(default_factory=list, description="A list of attendee email addresses.")
    description: Optional[str] = Field(None, description="A description for the event.")
    location: Optional[str] = Field(None, description="The location of the event.")

def create_calendar_event_func(
        summary: str,
        start_datetime_iso: str,
        end_datetime_iso: str,
        timezone: str,
        attendees: Optional[List[str]] = None,
        description: Optional[str] = None,
        location: Optional[str] = None
    ) -> Dict[str, Any]:
    service = get_calendar_service()
    if not service:
        return {"status": "error", "message": "Failed to get Calendar service."}

    processed_attendees = attendees if attendees is not None else []
    event_body = {
        'summary': summary, 'location': location, 'description': description,
        'start': {'dateTime': start_datetime_iso, 'timeZone': timezone},
        'end': {'dateTime': end_datetime_iso, 'timeZone': timezone},
        'attendees': [{'email': email_addr} for email_addr in processed_attendees if email_addr],
        'reminders': {'useDefault': False, 'overrides': [{'method': 'email', 'minutes': 24 * 60}, {'method': 'popup', 'minutes': 30}]},
    }
    try:
        created_event = service.events().insert(calendarId='primary', body=event_body, sendUpdates="all").execute()
        logging.info(f"Event created: {created_event.get('summary')} - {created_event.get('htmlLink')}")
        return {
            "status": "success", "message": f"Event created successfully! Summary: {created_event.get('summary')}",
            "eventId": created_event.get('id'), "htmlLink": created_event.get('htmlLink')
        }
    except HttpError as error:
        error_details = getattr(error.resp, 'get', lambda x,y: str(error))('x-debug-info', str(error))
        logging.error(f"HttpError creating calendar event: {error_details}", exc_info=True)
        return {"status": "error", "message": f"HttpError: {error_details}", "details_str": str(error)}
    except Exception as e:
        logging.error(f"Unexpected error creating calendar event: {str(e)}", exc_info=True)
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}

calendar_event_creator_tool = StructuredTool.from_function(
    func=create_calendar_event_func,
    name="CreateGoogleCalendarEvent",
    description="Use this tool to create a new event on the user's primary Google Calendar.",
    args_schema=CreateCalendarEventInput
)

class DeleteCalendarEventInput(BaseModel):
    event_id: str = Field(description="The unique ID of the calendar event to be deleted.")

def delete_calendar_event_func(event_id: str) -> Dict[str, Any]:
    service = get_calendar_service()
    if not service:
        return {"status": "error", "message": "Failed to get Calendar service."}

    if not event_id:
        return {"status": "error", "message": "Event ID must be provided to delete an event."}

    try:
        service.events().delete(
            calendarId='primary',
            eventId=event_id,
            sendUpdates='all'
        ).execute()
        logging.info(f"Event with ID: {event_id} deleted successfully.")
        return {
            "status": "success",
            "message": f"Event ID: {event_id} deleted successfully."
        }
    except HttpError as error:
        if error.resp.status == 404 or error.resp.status == 410:
            logging.warning(f"Attempted to delete event ID: {event_id}, but it was not found (already deleted?). Error: {error}")
            return {"status": "success", "message": f"Event ID: {event_id} was not found, likely already deleted."}
        error_details = getattr(error.resp, 'get', lambda x, y: str(error))('x-debug-info', str(error))
        logging.error(f"HttpError deleting calendar event {event_id}: {error_details}", exc_info=True)
        return {"status": "error", "message": f"HttpError: {error_details}", "details_str": str(error)}
    except Exception as e:
        logging.error(f"Unexpected error deleting calendar event {event_id}: {str(e)}", exc_info=True)
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}

calendar_event_deleter_tool = StructuredTool.from_function(
    func=delete_calendar_event_func,
    name="DeleteGoogleCalendarEvent",
    description="Use this tool to delete an existing event from the user's primary Google Calendar using its event ID.",
    args_schema=DeleteCalendarEventInput
)

# --- NEW UPDATE FUNCTIONALITY ---
class UpdateCalendarEventInput(BaseModel):
    event_id: str = Field(description="The unique ID of the calendar event to be updated.")
    summary: Optional[str] = Field(None, description="The new title or summary of the event. If None, summary is not changed.")
    start_datetime_iso: Optional[str] = Field(None, description="The new start date and time in ISO 8601 format. If None, start time is not changed.")
    end_datetime_iso: Optional[str] = Field(None, description="The new end date and time in ISO 8601 format. If None, end time is not changed.")
    timezone: Optional[str] = Field(None, description="The new timezone for the event (e.g., 'Asia/Beirut'). If None, timezone is not changed if start/end are also None.")
    attendees: Optional[List[str]] = Field(None, description="The new list of attendee email addresses. If None, attendees are not changed. Provide an empty list to remove all attendees.")
    description: Optional[str] = Field(None, description="The new description for the event. If None, description is not changed.")
    location: Optional[str] = Field(None, description="The new location of the event. If None, location is not changed.")

def update_calendar_event_func(
        event_id: str,
        summary: Optional[str] = None,
        start_datetime_iso: Optional[str] = None,
        end_datetime_iso: Optional[str] = None,
        timezone: Optional[str] = None,
        attendees: Optional[List[str]] = None,
        description: Optional[str] = None,
        location: Optional[str] = None
    ) -> Dict[str, Any]:
    service = get_calendar_service()
    if not service:
        return {"status": "error", "message": "Failed to get Calendar service."}

    if not event_id:
        return {"status": "error", "message": "Event ID must be provided to update an event."}

    event_patch_body = {}
    if summary is not None:
        event_patch_body['summary'] = summary
    if start_datetime_iso is not None and timezone is not None: # Timezone must be provided if start/end is changed
        event_patch_body['start'] = {'dateTime': start_datetime_iso, 'timeZone': timezone}
    if end_datetime_iso is not None and timezone is not None: # Timezone must be provided if start/end is changed
        event_patch_body['end'] = {'dateTime': end_datetime_iso, 'timeZone': timezone}
    if attendees is not None: # Send empty list to remove all, or new list to replace
        event_patch_body['attendees'] = [{'email': email_addr} for email_addr in attendees if email_addr]
    if description is not None:
        event_patch_body['description'] = description
    if location is not None:
        event_patch_body['location'] = location
    
    # If only timezone is provided without start/end, it might not be directly patchable this way
    # Google usually expects timezone as part of start/end objects.
    # If start/end are not changing, the event's original timezone usually persists unless explicitly moved via start/end.

    if not event_patch_body:
        return {"status": "info", "message": "No changes provided to update the event."}

    try:
        # Using patch to only update specified fields.
        # For a full update where all fields are replaced, service.events().update() would be used.
        updated_event = service.events().patch(
            calendarId='primary',
            eventId=event_id,
            body=event_patch_body,
            sendUpdates='all' # Notifies attendees about changes
        ).execute()
        logging.info(f"Event ID: {event_id} updated successfully. Summary: {updated_event.get('summary')}")
        return {
            "status": "success",
            "message": f"Event ID: {event_id} updated successfully! Summary: {updated_event.get('summary')}",
            "eventId": updated_event.get('id'),
            "htmlLink": updated_event.get('htmlLink')
        }
    except HttpError as error:
        error_details = getattr(error.resp, 'get', lambda x,y: str(error))('x-debug-info', str(error))
        logging.error(f"HttpError updating calendar event {event_id}: {error_details}", exc_info=True)
        return {"status": "error", "message": f"HttpError: {error_details}", "details_str": str(error)}
    except Exception as e:
        logging.error(f"Unexpected error updating calendar event {event_id}: {str(e)}", exc_info=True)
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}

calendar_event_updater_tool = StructuredTool.from_function(
    func=update_calendar_event_func,
    name="UpdateGoogleCalendarEvent",
    description="Use this tool to update an existing event on the user's primary Google Calendar using its event ID. Only provided fields will be updated.",
    args_schema=UpdateCalendarEventInput
)