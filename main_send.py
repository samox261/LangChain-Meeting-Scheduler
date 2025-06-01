from tools.email_tools import email_reader_tool, email_sender_tool # Import both tools
from dotenv import load_dotenv
import os 
from datetime import datetime # For the Paris time example
import pytz # For the Paris time example
from tools.nlp_tools import email_analyzer_tool

def main():
    load_dotenv() 

    print("Testing LangChain Email Tools...")

    # --- Test Email Reading (Optional - you can comment this part out if it works fine) ---
    # print("\n--- Testing Email Reader Tool ---")
    # try:
    #     # Example: Read 1 email (input is a string)
    #     email_results = email_reader_tool.invoke("1") 
    #     if isinstance(email_results, str):
    #         print(email_results)
    #     else:
    #         for email in email_results:
    #             print(f"\n[Reader Tool] Subject: {email.get('subject')}")
    #             print(f"[Reader Tool] From: {email.get('from')}")
    #             print(f"[Reader Tool] Snippet: {email.get('snippet')}")
    # except Exception as e:
    #     print(f"Error reading email with tool: {e}")

    # --- Test Email Sending ---
    print("\n--- Testing Email Sender Tool ---")
    
    # IMPORTANT: Replace with an actual email address for testing!
    recipient_email = "cristina.marquigny@gmail.com"  # <<< CHANGE THIS!!!
    
    email_subject = "Test Email from my LangChain Agent!"
    email_body = f"Hello from Paris!\n\nThis is a test email sent by my new AI assistant.\nHope you're having a great day around {os.environ.get('CURRENT_TIME_PARIS', 'this time')}! \n\nBest,\nFawaz"
    
    tool_input_for_sending = {
        "to_address": recipient_email,
        "subject": email_subject,
        "message_text": email_body
    }

    # Safety check against the placeholder email
    if recipient_email == "YOUR_TEST_RECIPIENT_EMAIL@example.com": 
        print("\nIMPORTANT: Please update the 'recipient_email' in your test script before running the send test.")
    else:
        try:
            print(f"Attempting to send email to: {recipient_email}")
            send_result = email_sender_tool.invoke(tool_input_for_sending)
            print(f"Send Result: {send_result}")
        except Exception as e:
            print(f"An error occurred while trying to send the email with the tool: {e}")

if __name__ == '__main__':
    # Set Beirut time environment variable for the email body
    try:
        beirut_tz = pytz.timezone("Asia/Beirut")
        beirut_time = datetime.now(beirut_tz).strftime("%I:%M %p %Z")
        os.environ['CURRENT_TIME_BEIRUT'] = beirut_time
    except Exception as e:
        print(f"Could not set Beirut time: {e}") # In case pytz or datetime fails for some reason
        os.environ['CURRENT_TIME_BEIRUT'] = "the current time"
        
    main()