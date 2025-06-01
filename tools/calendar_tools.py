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

def get_calendar_service() -> Optional[build]: # Does not take user_email
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

def create_calendar_event_func( # Name without _func, does not take user_email
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
    func=create_calendar_event_func, # Uses create_calendar_event_func
    name="CreateGoogleCalendarEvent",
    description="Use this tool to create a new event on the user's primary Google Calendar.",
    args_schema=CreateCalendarEventInput
)
