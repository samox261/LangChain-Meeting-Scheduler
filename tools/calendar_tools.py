import os.path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any # Ensure all are imported

# Google Auth and API client libraries
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# LangChain Tool imports
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

# --- Configuration ---
TOKEN_FILE = 'token.json'
CREDENTIALS_FILE = 'credentials.json'

# --- Calendar Service Authentication & Functions ---
def get_calendar_service():
    creds = None
    current_scopes_for_calendar = ['https://www.googleapis.com/auth/calendar']
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, current_scopes_for_calendar)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"Error refreshing token for calendar: {e}. Re-authenticating.")
                creds = None
        if not creds:
            print("Attempting to re-authenticate for calendar service (token missing, invalid, or refresh failed)...")
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE, current_scopes_for_calendar) # Use all combined scopes from SCOPES in email_tools.py
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    try:
        service = build('calendar', 'v3', credentials=creds)
        return service
    except HttpError as error:
        print(f'An error occurred while building the calendar service: {error}')
        return None

class CreateCalendarEventInput(BaseModel):
    summary: str = Field(description="The title or summary of the event.")
    start_datetime_iso: str = Field(description="The start date and time in ISO 8601 format.")
    end_datetime_iso: str = Field(description="The end date and time in ISO 8601 format.")
    attendees: Optional[List[str]] = Field(default_factory=list, description="A list of attendee email addresses.")
    description: Optional[str] = Field(None, description="A description for the event.")
    location: Optional[str] = Field(None, description="The location of the event.")
    timezone: str = Field(description="The timezone for the event (e.g., 'Europe/Paris').")

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
        return {"status": "error", "message": "Failed to authenticate or build Calendar service."}

    processed_attendees = attendees if attendees is not None else []
    event_body = {
        'summary': summary,
        'location': location,
        'description': description,
        'start': {'dateTime': start_datetime_iso, 'timeZone': timezone},
        'end': {'dateTime': end_datetime_iso, 'timeZone': timezone},
        'attendees': [{'email': email} for email in processed_attendees],
        'reminders': {
            'useDefault': False,
            'overrides': [
                {'method': 'email', 'minutes': 24 * 60},
                {'method': 'popup', 'minutes': 30},
            ],
        },
    }
    try:
        created_event = service.events().insert(calendarId='primary', body=event_body, sendUpdates="all").execute()
        return {
            "status": "success",
            "message": f"Event created successfully! Link: {created_event.get('htmlLink')}",
            "eventId": created_event.get('id'),
            "htmlLink": created_event.get('htmlLink')
        }
    except HttpError as error:
        return {"status": "error", "message": f"An error occurred creating calendar event: {error}"}
    except Exception as e:
        return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}

calendar_event_creator_tool = StructuredTool.from_function(
    func=create_calendar_event_func,
    name="CreateGoogleCalendarEvent",
    description="Use this tool to create a new event on the user's primary Google Calendar.",
    args_schema=CreateCalendarEventInput
)