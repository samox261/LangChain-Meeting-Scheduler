from tools.email_tools import email_reader_tool # Import the tool
from dotenv import load_dotenv

def main():
    load_dotenv() 

    print("Testing the LangChain Email Reader Tool...")

    try:
        # To run the tool, you call its .invoke() method.
        # Pass the input as a string, as the tool description suggests.
        tool_input_str = "2"  # Get 2 recent emails
        print(f"Invoking tool with input: '{tool_input_str}'")
        email_results = email_reader_tool.invoke(tool_input_str) # Pass the string
        
        # You can also test the default behavior (should fetch 5 emails)
        # print("\nInvoking tool with no input (should use default)...")
        # email_results_default = email_reader_tool.invoke({}) # or email_reader_tool.invoke(None)
        # print(email_results_default) # You'd format this nicely too

        print("\n--- Results from Email Reader Tool ---")
        if isinstance(email_results, str): # Error message or "no messages"
            print(email_results)
        elif email_results:
            for email in email_results:
                print(f"\n--- Email ---")
                print(f"ID: {email.get('id')}")
                print(f"From: {email.get('from')}")
                print(f"Subject: {email.get('subject')}")
                print(f"Date: {email.get('date')}")
                print(f"Snippet: {email.get('snippet')}")
                # print(f"Body:\n{email.get('body_text')[:200]}...") # Uncomment to see body
                print("------------------------------------")
        else:
            print("Tool returned no results or an unexpected empty response.")

    except Exception as e:
        print(f"An error occurred while running the tool: {e}")

if __name__ == '__main__':
    main()