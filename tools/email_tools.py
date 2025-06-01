import os.path
import base64
from typing import Union, Optional, List, Dict, Any
from email.utils import getaddresses
import logging

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from langchain_core.tools import Tool
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from email.mime.text import MIMEText

# --- Configuration ---
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/gmail.modify',
    'https://www.googleapis.com/auth/calendar'
]
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.json' # Single token file for the agent

# --- Gmail Service Authentication ---
def get_gmail_service() -> Optional[build]:
    creds = None
    if os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except Exception as e:
            logging.warning(f"Could not load token from {TOKEN_FILE} (may need re-auth or scopes changed): {e}")
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                logging.info(f"Refreshing token for Gmail service...")
                creds.refresh(Request())
                logging.info(f"Token refreshed successfully.")
            except Exception as e:
                logging.warning(f"Error refreshing token: {e}. Will attempt re-authentication.")
                creds = None

        if not creds:
            logging.info(f"Initiating new OAuth flow for Gmail service (scopes: {SCOPES})...")
            if not os.path.exists(CREDENTIALS_FILE):
                logging.error(f"CRITICAL: {CREDENTIALS_FILE} not found. Cannot initiate OAuth flow.")
                return None
            try:
                flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
                creds = flow.run_local_server(port=0)
            except Exception as e_flow:
                logging.error(f"OAuth flow failed: {e_flow}", exc_info=True)
                return None

        if creds:
            with open(TOKEN_FILE, 'w') as token_file_handle:
                token_file_handle.write(creds.to_json())
            logging.info(f"Credentials saved to {TOKEN_FILE}")
        else:
            logging.error(f"Could not obtain credentials after OAuth flow.")
            return None

    try:
        service = build('gmail', 'v1', credentials=creds)
        logging.debug("Gmail service client built successfully.")
        return service
    except HttpError as error:
        logging.error(f'An HttpError occurred while building the Gmail service: {error}')
        return None
    except Exception as e:
        logging.error(f'An unexpected error occurred in get_gmail_service: {str(e)}', exc_info=True)
        return None

# --- Email Reading Functionality ---
def read_recent_emails(max_results: int = 5) -> Union[List[Dict[str, Any]], str]:
    service = get_gmail_service()
    if not service:
        return "Failed to get Gmail service for reading."
    try:
        num_max_results = int(max_results)
        # MODIFIED: Fetch recent messages from INBOX, regardless of read status by the user
        logging.debug(f"Attempting to fetch up to {num_max_results} recent emails from INBOX...")
        results = service.users().messages().list(userId='me', labelIds=['INBOX'], maxResults=num_max_results).execute()
        messages = results.get('messages', [])

        if not messages:
            return f"No messages found in INBOX matching criteria (max_results={num_max_results})."

        email_list = []
        logging.debug(f"Found {len(messages)} message(s) to process.")
        for message_info in messages:
            msg = service.users().messages().get(userId='me', id=message_info['id']).execute()
            email_data = { 'id': msg.get('id'), 'threadId': msg.get('threadId'), 'snippet': msg.get('snippet', '')}
            headers = msg.get('payload', {}).get('headers', [])
            from_details = {"name": "", "email": ""}
            cc_recipients_list = []
            for header in headers:
                name = header.get('name', '').lower(); value = header.get('value', '')
                if name == 'subject': email_data['subject'] = value
                elif name == 'from':
                    parsed_from = getaddresses([value]);
                    if parsed_from: from_details['name'], from_details['email'] = parsed_from[0]
                elif name == 'date': email_data['date'] = value
                elif name == 'cc':
                    parsed_cc = getaddresses([value])
                    for p_name, p_email_addr in parsed_cc:
                        if p_email_addr: cc_recipients_list.append({"name": p_name, "email": p_email_addr})
            email_data['from_details'] = from_details
            email_data['cc_recipients'] = cc_recipients_list
            payload = msg.get('payload', {}); body_text = ""
            if 'parts' in payload:
                for part in payload['parts']:
                    if part.get('mimeType') == 'text/plain':
                        encoded_body = part.get('body', {}).get('data', '')
                        if encoded_body: body_text += base64.urlsafe_b64decode(encoded_body).decode('utf-8')
                        break
            else:
                encoded_body = payload.get('body', {}).get('data', '')
                if encoded_body: body_text = base64.urlsafe_b64decode(encoded_body).decode('utf-8')
            email_data['body_text'] = body_text.strip()
            email_list.append(email_data)
        return email_list
    except HttpError as error: return f"HttpError reading emails: {error}"
    except Exception as e: logging.error(f"Unexpected error in read_recent_emails: {str(e)}", exc_info=True); return f"Unexpected error: {str(e)}"

def _parse_input_for_read_emails(tool_input: Union[str, int, dict, None] = None) -> int:
    default_max_results = 5;
    if isinstance(tool_input, dict): input_val = tool_input.get('max_results', default_max_results)
    else: input_val = tool_input
    if input_val is None: return default_max_results
    try: num_emails = int(input_val); return num_emails if num_emails > 0 else default_max_results
    except (ValueError, TypeError): logging.warning(f"Invalid input '{tool_input}' for max_results. Using default."); return default_max_results

email_reader_tool = Tool(
    name="ReadRecentGmailEmails",
    func=lambda tool_input=None: read_recent_emails(max_results=_parse_input_for_read_emails(tool_input)),
    description="Use this tool to read a specified number of the most recent emails from the user's Gmail inbox. Input can be a string or integer for the maximum number of emails (e.g., \"3\" or 3 or {\"max_results\": 3}). If no input or invalid input, defaults to fetching 5 emails. Returns a list of email details or an error string." # MODIFIED description slightly
)

def send_gmail_email(to_address: str, subject: str, message_text: str) -> str:
    service = get_gmail_service()
    if not service: return "Failed to get Gmail service for sending."
    try:
        mime_message = MIMEText(message_text)
        mime_message['to'] = to_address; mime_message['subject'] = subject
        encoded_message = base64.urlsafe_b64encode(mime_message.as_bytes()).decode()
        create_message_body = {'raw': encoded_message}
        send_message_response = service.users().messages().send(userId="me", body=create_message_body).execute()
        return f"Email sent successfully! Message ID: {send_message_response.get('id')}"
    except HttpError as error: return f"HttpError sending email: {error}"
    except Exception as e: logging.error(f"Unexpected error sending email: {str(e)}", exc_info=True); return f"Unexpected error: {str(e)}"

class SendEmailInput(BaseModel):
    to_address: str = Field(description="The email address of the recipient.")
    subject: str = Field(description="The subject of the email.")
    message_text: str = Field(description="The plain text body of the email.")

email_sender_tool = StructuredTool.from_function(
    func=send_gmail_email,
    name="SendGmailEmail",
    description="Use this tool to send an email. You must provide the recipient's email address ('to_address'), the email 'subject', and the 'message_text' (body of the email).",
    args_schema=SendEmailInput
)

def mark_email_as_read_func(message_id: str) -> str:
    service = get_gmail_service()
    if not service: return "Failed to get Gmail service for marking as read."
    try:
        # This removes the 'UNREAD' label, effectively marking it as read.
        service.users().messages().modify(userId='me', id=message_id, body={'removeLabelIds': ['UNREAD']}).execute()
        return f"Successfully marked message ID {message_id} as read."
    except HttpError as error: return f"HttpError marking email as read: {error}"
    except Exception as e: logging.error(f"Unexpected error marking email as read: {str(e)}", exc_info=True); return f"Unexpected error: {str(e)}"

class MarkAsReadInput(BaseModel):
    message_id: str = Field(description="The unique ID of the email message to be marked as read.")

mark_as_read_tool = StructuredTool.from_function(
    func=mark_email_as_read_func,
    name="MarkEmailAsRead",
    description="Use this tool to mark an email as read in Gmail once it has been processed. Requires the 'message_id'.",
    args_schema=MarkAsReadInput
)