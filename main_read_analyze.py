from dotenv import load_dotenv
import os
import json # For pretty-printing JSON
from datetime import datetime 
import pytz 

# Import your tools
from tools.email_tools import email_reader_tool, email_sender_tool 
from tools.nlp_tools import email_analyzer_tool

def main():
    load_dotenv() 

    print("--- Testing Email Reading and Analysis ---")
    num_emails_to_read_and_analyze = "1" # Read 1 email for this test

    try:
        print(f"\nStep 1: Reading {num_emails_to_read_and_analyze} recent email(s)...")
        # Invoke the email reader tool
        # The input should be a string for this tool as per its lambda wrapper
        email_list_result = email_reader_tool.invoke(num_emails_to_read_and_analyze)

        if isinstance(email_list_result, str): # Error message from reader tool
            print(f"Email Reader Tool Error: {email_list_result}")
            return
        
        if not email_list_result:
            print("No emails found by the Email Reader Tool.")
            return

        print(f"Successfully read {len(email_list_result)} email(s).")

        # Analyze the first email fetched (or loop through them if you like)
        for i, email_data in enumerate(email_list_result):
            print(f"\nStep 2: Analyzing Email #{i+1}")
            email_subject = email_data.get("subject", "No Subject Provided")
            email_body = email_data.get("body_text", "No Body Provided")

            print(f"  Subject: {email_subject}")
            print(f"  Body (first 100 chars): {email_body[:100]}...")

            if not email_body.strip() and not email_subject.strip():
                print("  Email has no subject or body content to analyze. Skipping.")
                continue

            analyzer_input = {
                "email_subject": email_subject,
                "email_body": email_body
            }
            
            try:
                analysis_result = email_analyzer_tool.invoke(analyzer_input)
                print("\n  Analysis Result (JSON):")
                if isinstance(analysis_result, dict):
                    print(json.dumps(analysis_result, indent=2))
                else:
                    # This might happen if analyze_email_content itself returns an error string
                    print(analysis_result) 
            except Exception as e_analyze:
                print(f"  Error during email analysis: {e_analyze}")
            print("-" * 30)

    except Exception as e_read:
        print(f"An error occurred while trying to read emails with the tool: {e_read}")

    # --- Optional: Test Email Sending (uncomment if you want to test this too) ---
    # print("\n--- Testing Email Sender Tool (Currently Commented Out) ---")
    # recipient_email = "YOUR_TEST_RECIPIENT_EMAIL@example.com"  # <<< CHANGE THIS if uncommenting!
    # email_subject_send = "Test Email from my LangChain Agent!"
    # email_body_send = f"Hello from Paris!\n\nThis is a test email sent by my new AI assistant.\nHope you're having a great day around {os.environ.get('CURRENT_TIME_PARIS', 'this time')}! \n\nBest,\nFawaz"
    # tool_input_for_sending = {
    #     "to_address": recipient_email,
    #     "subject": email_subject_send,
    #     "message_text": email_body_send
    # }
    # if recipient_email == "YOUR_TEST_RECIPIENT_EMAIL@example.com": 
    #     print("\nIMPORTANT: Please update 'recipient_email' if uncommenting the send test.")
    # else:
    #     try:
    #         print(f"Attempting to send email to: {recipient_email}")
    #         send_result = email_sender_tool.invoke(tool_input_for_sending)
    #         print(f"Send Result: {send_result}")
    #     except Exception as e_send:
    #         print(f"An error occurred while trying to send email: {e_send}")

if __name__ == '__main__':
    try:
        paris_tz = pytz.timezone("Europe/Paris")
        paris_time = datetime.now(paris_tz).strftime("%I:%M %p %Z")
        os.environ['CURRENT_TIME_PARIS'] = paris_time
    except Exception as e_time:
        print(f"Could not set Paris time: {e_time}")
        os.environ['CURRENT_TIME_PARIS'] = "the current time"
    main()