import json
from typing import Optional, List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from datetime import datetime # For current_datetime_for_llm
import pytz # For current_datetime_for_llm

# --- Pydantic Models ---
class MeetingDetails(BaseModel):
    intent: Optional[str] = Field(None, description="The classified scheduling intent.")
    attendees: Optional[List[str]] = Field(default_factory=list, description="List of attendees (valid email addresses if possible).")
    topic: Optional[str] = Field(None, description="Meeting topic.")
    proposed_dates_times: Optional[List[str]] = Field(default_factory=list, description="List of proposed dates/times as strings.")
    duration_minutes: Optional[int] = Field(None, description="Meeting duration in minutes.")
    constraints_preferences: Optional[str] = Field(None, description="Other constraints or preferences.")

class AnalyzeEmailInput(BaseModel):
    """Input schema for EmailContentAnalyzer tool."""
    email_subject: str = Field(description="The subject line of the email.")
    email_body: str = Field(description="The main body text of the email.")
    cc_recipient_emails: Optional[List[str]] = Field(default_factory=list, description="A list of email addresses from the email's CC field.")
    conversation_context: Optional[str] = Field(None, description="A summary of the previous conversation in this email thread, if any.")

# --- LLM-based Functions ---
def analyze_email_content(
    email_subject: str,
    email_body: str,
    cc_recipient_emails: Optional[List[str]] = None,
    conversation_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyzes email subject, body, CC list, and conversation context using an LLM
    to extract meeting scheduling details for the CURRENT email.
    Returns a dictionary with extracted information.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, convert_system_message_to_human=True)

    try:
        user_timezone_str = "Europe/Paris" # Ideally, pass this in or get from shared config
        user_tz = pytz.timezone(user_timezone_str)
        current_datetime_for_llm_val = datetime.now(user_tz).strftime("%A, %B %d, %Y, %I:%M %p %Z (%z)")
    except Exception:
        current_datetime_for_llm_val = datetime.now().strftime("%A, %B %d, %Y, %I:%M %p") # Fallback

    cc_list_str_val = ", ".join(cc_recipient_emails) if cc_recipient_emails and len(cc_recipient_emails) > 0 else "None"
    conversation_context_for_prompt_val = conversation_context if conversation_context and conversation_context.strip() else "No previous conversation context available for this scheduling attempt. This is the first interaction being analyzed for this task."

    prompt_template = """You are an expert administrative assistant AI. Your task is to meticulously analyze the CURRENT email provided (subject and body) to understand its primary purpose, especially concerning meeting scheduling from the perspective of the recipient.
Today's date and current time for your reference is: {current_datetime_for_llm}.

You are also provided with a summary of the previous conversation context for this email thread, if available. Use this context to better understand the CURRENT email, but your final JSON output should pertain ONLY to the information and intent explicitly derivable from the CURRENT email.

--- CONVERSATION CONTEXT START ---
{conversation_context_for_prompt}
--- CONVERSATION CONTEXT END ---

Now, analyze the CURRENT email below:

CURRENT Email Subject:
---
{email_subject}
---

CURRENT Email Body:
---
{email_body}
---

CC'd List in CURRENT email:
---
{cc_list_str_for_prompt}
---

Respond ONLY with a valid JSON object detailing the analysis of the CURRENT email, using the following keys:
- "intent": (String) Classify the primary intent of the CURRENT email. Consider the conversation context.
    Possible values: "schedule_new_meeting", "reschedule_meeting", "cancel_meeting", "confirm_attendance", "decline_attendance", "propose_new_time", "meeting_related_query", "not_meeting_related".
    - "schedule_new_meeting": Use for any initial proposal, request, or assertive statement to set up a NEW meeting.
    - "reschedule_meeting": If the email explicitly requests to change a PREVIOUSLY AGREED-UPON or SCHEDULED meeting.
    - "propose_new_time": If, during an ONGOING dialogue (see context), alternative times are suggested because prior proposals failed.
    - "confirm_attendance": Sender confirms THEIR attendance, or asks RECIPIENT to confirm for a known event.
    - "decline_attendance": Sender declines an invitation.
    - "cancel_meeting": Explicitly cancels a scheduled meeting.
    - "meeting_related_query": Questions about logistics/agenda of a known meeting.
    - "not_meeting_related": If the CURRENT email is not about scheduling.

- "attendees": (List of Strings or null) From the CURRENT email, list ONLY VALID EMAIL ADDRESSES of individuals (excluding the agent/recipient itself unless explicitly stated for a meeting they are just attending) who should be part of the meeting.
    - If only a name is mentioned for a potential attendee in the email body, and no email address for them is apparent, do NOT include that name. Prioritize explicit email addresses found or directly implied for attendees.
    - Regarding the CC'd Recipients provided (CC'd List: {cc_list_str_for_prompt}): ONLY include their email addresses IF the CURRENT email body contains a clear, explicit instruction to include them in the *meeting itself* (e.g., "Please include everyone on CC in the meeting," "Inviting the CC'd folks as well," "Looping in those CC'd for the discussion").
    - (Agent logic will handle adding the original email sender of the thread separately where appropriate).
    If no valid email addresses for attendees (other than the sender/recipient which are handled by agent logic) are identified, use null or an empty list.

- "topic": (String or null) The main topic from the CURRENT email. If context mentions a topic and the current email doesn't specify a new one, you may infer it if appropriate.
- "proposed_dates_times": (List of Strings or null) From the CURRENT email, list only the NEWLY SUGGESTED dates and times for the meeting.
    - If the email is a reschedule request or proposes alternatives, focus *only* on extracting the NEW alternative times proposed in THIS email.
    - If it's an initial meeting request, extract all proposed times from THIS email.
    If no new times are clearly proposed, use null or an empty list.

- "duration_minutes": (Integer or null) From the CURRENT email, the proposed duration in minutes.
- "constraints_preferences": (String or null) From the CURRENT email, any other relevant constraints or preferences.

Ensure the output is a single, complete, and valid JSON object.

JSON Output:
"""
    prompt = prompt_template.format(
        current_datetime_for_llm=current_datetime_for_llm_val,
        conversation_context_for_prompt=conversation_context_for_prompt_val,
        email_subject=email_subject,
        email_body=email_body,
        cc_list_str_for_prompt=cc_list_str_val
    )

    try:
        response = llm.invoke(prompt)
        content_string = response.content
        
        if content_string.startswith("```json"):
            content_string = content_string[len("```json"):].strip()
        if content_string.endswith("```"):
            content_string = content_string[:-len("```")].strip()
        
        parsed_json = json.loads(content_string.strip())
        
        try:
            validated_details = MeetingDetails(**parsed_json)
            return validated_details.model_dump()
        except Exception as val_error:
            print(f"Pydantic validation error for Email Analysis: {val_error} on LLM output: {parsed_json}")
            parsed_json["pydantic_validation_error"] = str(val_error)
            return parsed_json

    except json.JSONDecodeError as e:
        raw_c = content_string if 'content_string' in locals() else "Content string error"
        print(f"JSONDecodeError in Email Analysis: {e} - Raw LLM output was: '{raw_c}'")
        return {"error": "Failed to decode JSON response from LLM.", "raw_response": raw_c}
    except Exception as e:
        raw_resp_c = getattr(response, 'content', "No content in resp obj.") if 'response' in locals() else "No resp obj."
        print(f"An error occurred during Email Analysis LLM call: {e} - Raw LLM output: '{raw_resp_c}'")
        return {"error": f"An error occurred during Email Analysis LLM call: {str(e)}", "raw_response": raw_resp_c}


def normalize_datetime_with_llm(
    natural_datetime_str: str,
    reference_datetime_iso: str,
    target_timezone_str: str
) -> Optional[str]:
    if not natural_datetime_str:
        print("  [LLM Date Normalizer] Received empty natural_datetime_str.")
        return None
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0, convert_system_message_to_human=True)
    prompt_template = """Your sole task is to convert a natural language date and time string into a "YYYY-MM-DDTHH:MM:SS" format.
Reference Information:
- Current date and time: {reference_datetime_iso}
- Target timezone for the output: {target_timezone_str}
- Natural language input: "{natural_datetime_str}"

Convert the "Natural language input" into the "YYYY-MM-DDTHH:MM:SS" string format, considering the current reference date and time and the target timezone.
Ensure the output strictly adheres to this format.
If a precise conversion to "YYYY-MM-DDTHH:MM:SS" is impossible due to ambiguity (e.g., "sometime next week" without a specific day or time), or if the input is not a date/time, output the exact string "None".
Do not include any other text, explanations, or markdown.

Converted Datetime:"""
    prompt = prompt_template.format(
        natural_datetime_str=natural_datetime_str,
        reference_datetime_iso=reference_datetime_iso,
        target_timezone_str=target_timezone_str
    )
    try:
        response = llm.invoke(prompt)
        raw_content_from_llm = response.content
        print(f"  [LLM Date Normalizer] Raw response for '{natural_datetime_str}': '{raw_content_from_llm}'")
        normalized_str = raw_content_from_llm.strip()
        if normalized_str.lower() == "none" or not normalized_str:
            print(f"  LLM indicated it could not normalize '{natural_datetime_str}'.")
            return None
        if len(normalized_str) == 19 and normalized_str[4] == '-' and normalized_str[7] == '-' and \
           normalized_str[10] == 'T' and normalized_str[13] == ':' and normalized_str[16] == ':':
            try:
                datetime.strptime(normalized_str, '%Y-%m-%dT%H:%M:%S') # Check parsability
                return normalized_str
            except ValueError:
                print(f"  LLM returned a string '{normalized_str}' that looks like ISO but is not a valid datetime.")
                return None
        else:
            print(f"  LLM returned an unexpected format: '{normalized_str}' for input '{natural_datetime_str}'. Expected YYYY-MM-DDTHH:MM:SS.")
            return None
    except Exception as e:
        print(f"  Error during LLM datetime normalization for '{natural_datetime_str}': {e}")
        return None

email_analyzer_tool = StructuredTool.from_function(
    func=analyze_email_content,
    name="EmailContentAnalyzer",
    description="Analyzes the subject, body, CC list, and conversation context of an email to identify scheduling intent and extract meeting details. Returns a JSON object.",
    args_schema=AnalyzeEmailInput,
    # return_schema=MeetingDetails # Optional: Enforce output structure
)