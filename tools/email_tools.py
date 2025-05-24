import os.path
import base64
from typing import Union, Optional, List, Dict, Any # Ensure all typing imports
from email.utils import getaddresses # For parsing Cc and From headers

# Google Auth and API client libraries
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# LangChain Tool imports
from langchain_core.tools import Tool
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

# For creating email messages
from email.mime.text import MIMEText

# --- Configuration ---
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/calendar'
]
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.json'

# --- Gmail Service Authentication ---
def get_gmail_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"Error refreshing token: {e}. Re-authenticating.")
                creds = None # Force re-authentication
        
        if not creds: # If still no creds after potential refresh attempt
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    try:
        service = build('gmail', 'v1', credentials=creds)
        return service
    except HttpError as error:
        print(f'An error occurred while building the Gmail service: {error}')
        return None

# --- Email Reading Functionality ---
def read_recent_emails(max_results: int = 5) -> Union[List[Dict[str, Any]], str]:
    """
    Reads a specified number of recent emails from the user's Gmail inbox.
    Prioritizes unread emails, then falls back to most recent if no unread found (for max_results=1).
    Returns a list of dictionaries, where each dictionary contains:
    'id', 'threadId', 'snippet', 'subject', 'from_details', 'cc_recipients', 'date', and 'body_text'.
    'from_details': {'name': str, 'email': str}
    'cc_recipients': list of {'name': str, 'email': str}
    """
    service = get_gmail_service()
    if not service:
        return "Failed to authenticate or build Gmail service for reading."

    try:
        # Try to fetch UNREAD messages first
        print(f"Attempting to fetch up to {max_results} unread emails...")
        # Ensure max_results is an int for comparison later if needed
        num_max_results = int(max_results) 
        
        results = service.users().messages().list(userId='me', labelIds=['INBOX'], q='is:unread', maxResults=num_max_results).execute()
        messages = results.get('messages', [])

        # Fallback for the specific case where we ask for 1 email and no unread are found
        if not messages and num_max_results == 1:
            print(f"No unread messages found. Fetching the most recent message overall instead (max_results={num_max_results}).")
            results = service.users().messages().list(userId='me', labelIds=['INBOX'], maxResults=1).execute()
            messages = results.get('messages', [])

        email_list = []
        if not messages:
            # This message will now only appear if still no messages after fallback (if fallback was triggered)
            # Or if max_results > 1 and no unread messages were found.
            return f"No messages found matching criteria (max_results={num_max_results})." 
        
        print(f"Found {len(messages)} message(s) to process.")
        for message_info in messages:
            msg = service.users().messages().get(userId='me', id=message_info['id']).execute()
            email_data = {
                'id': msg['id'], 
                'threadId': msg['threadId'], 
                'snippet': msg['snippet']
            }
            headers = msg.get('payload', {}).get('headers', [])
            
            from_details = {"name": "", "email": ""}
            cc_recipients = []

            for header in headers:
                name = header.get('name', '').lower()
                value = header.get('value', '')
                if name == 'subject':
                    email_data['subject'] = value
                elif name == 'from':
                    parsed_from = getaddresses([value])
                    if parsed_from:
                        from_details['name'], from_details['email'] = parsed_from[0]
                elif name == 'date':
                    email_data['date'] = value
                elif name == 'cc':
                    parsed_cc = getaddresses([value])
                    for p_name, p_email_addr in parsed_cc:
                        cc_recipients.append({"name": p_name, "email": p_email_addr})
            
            email_data['from_details'] = from_details
            email_data['cc_recipients'] = cc_recipients

            payload = msg.get('payload', {})
            body_text = ""
            if 'parts' in payload:
                for part in payload['parts']:
                    if part['mimeType'] == 'text/plain':
                        encoded_body = part.get('body', {}).get('data', '')
                        if encoded_body:
                            body_text += base64.urlsafe_b64decode(encoded_body).decode('utf-8')
                        break 
            else: 
                encoded_body = payload.get('body', {}).get('data', '')
                if encoded_body:
                    body_text = base64.urlsafe_b64decode(encoded_body).decode('utf-8')
            
            email_data['body_text'] = body_text.strip()
            email_list.append(email_data)
            
        return email_list

    except HttpError as error:
        return f"An error occurred while reading emails: {error}"
    except Exception as e:
        import traceback # For more detailed error in this specific function
        print(f"Unexpected error in read_recent_emails: {traceback.format_exc()}")
        return f"An unexpected error occurred in read_recent_emails: {str(e)}"


def _parse_input_for_read_emails(tool_input: Union[str, int, dict, None] = None) -> int:
    default_max_results = 5 # Default if not specified or invalid
    # Allow passing a dictionary like {"max_results": 3} or just an int/str
    if isinstance(tool_input, dict):
        input_val = tool_input.get('max_results', default_max_results)
    else:
        input_val = tool_input

    if input_val is None:
        return default_max_results
    try:
        num_emails = int(input_val)
        return num_emails if num_emails > 0 else default_max_results
    except (ValueError, TypeError):
        print(f"Warning: Email reader tool received invalid input '{tool_input}'. Using default {default_max_results}.")
        return default_max_results

email_reader_tool = Tool(
    name="ReadRecentGmailEmails",
    func=lambda tool_input=None: read_recent_emails(max_results=_parse_input_for_read_emails(tool_input)),
    description="""Use this tool to read a specified number of the most recent emails 
                   from the user's Gmail inbox (prioritizes unread). 
                   Input can be a string or integer for the maximum number of emails (e.g., "3" or 3).
                   If no input or invalid input, defaults to fetching 5 emails.
                   Returns a list of email details or an error string.
                """
)

# --- Email Sending Functionality ---
def send_gmail_email(to_address: str, subject: str, message_text: str):
    service = get_gmail_service()
    if not service:
        return "Failed to authenticate or build Gmail service for sending."
    try:
        mime_message = MIMEText(message_text)
        mime_message['to'] = to_address
        mime_message['subject'] = subject
        encoded_message = base64.urlsafe_b64encode(mime_message.as_bytes()).decode()
        create_message = {'raw': encoded_message}
        send_message_response = service.users().messages().send(userId="me", body=create_message).execute()
        return f"Email sent successfully! Message ID: {send_message_response.get('id')}"
    except HttpError as error:
        return f"An error occurred while sending email: {error}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

class SendEmailInput(BaseModel):
    to_address: str = Field(description="The email address of the recipient.")
    subject: str = Field(description="The subject of the email.")
    message_text: str = Field(description="The plain text body of the email.")

email_sender_tool = StructuredTool.from_function(
    func=send_gmail_email,
    name="SendGmailEmail",
    description="Use this tool to send an email. Provide the recipient's email address, the subject, and the message body.",
    args_schema=SendEmailInput
)