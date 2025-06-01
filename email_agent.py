import re
from typing import Dict, Optional
from datetime import datetime
from enum import Enum
from email.message import EmailMessage
import os


class Intent(Enum):
    COMPLAINT = "complaint"
    REGULATION = "regulation"
    INVITE = "invite"
    RFQ = "RFQ"
    OTHER = "other"

class Urgency(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class EmailAgent:
    def __init__(self, intent_classifier=None):
        self.intent_classifier = intent_classifier

    def extract_sender(self, email_content: str) -> Optional[str]:
        """Extracts sender email from common email patterns"""
        from_match = re.search(r'From:\s*"?([^"<]+)"?\s*<([^>]+)>', email_content)
        if from_match:
            return from_match.group(2).strip()

        fallback_match = re.search(r'[\w\.-]+@[\w\.-]+', email_content)
        return fallback_match.group(0) if fallback_match else None

    def detect_urgency(self, email_content: str) -> Urgency:
        """Determines urgency based on content analysis"""
        content_lower = email_content.lower()

        if any(word in content_lower for word in ["urgent", "immediate", "asap", "time-sensitive"]):
            return Urgency.HIGH

        if any(word in content_lower for word in ["follow up", "request", "please respond"]):
            return Urgency.MEDIUM

        return Urgency.LOW

    def extract_body(self, email_content: str) -> str:
        """Removes headers and returns body text"""
        split_content = re.split(r'\n\s*\n', email_content, maxsplit=1)
        return split_content[1].strip() if len(split_content) > 1 else email_content.strip()

    

    def process_email(self, email_content) -> Dict:
        """
        Processes email content (plain text or EmailMessage) and extracts key information.
        """
        # If input is bytes, parse it
    
    # Handle file paths
        if isinstance(email_content, str) and os.path.isfile(email_content):
            with open(email_content, "rb") as f:
                email_content = f.read()
                
        if isinstance(email_content, bytes):
            from email import policy
            from email.parser import BytesParser
            email_obj = BytesParser(policy=policy.default).parsebytes(email_content)
        elif isinstance(email_content, EmailMessage):
            email_obj = email_content
        else:
            email_obj = None


        if email_obj:
            # Extract sender
            sender = email_obj.get('From')
            # Extract subject
            subject = email_obj.get('Subject', '')
            # Extract body (plain text)
            if email_obj.is_multipart():
                body = ''
                for part in email_obj.walk():
                    if part.get_content_type() == 'text/plain':
                        body += part.get_content()
            else:
                body = email_obj.get_content()
            # Use subject + body for intent/urgency
            text_for_analysis = f"{subject}\n{body}"
        else:
            # Fallback: treat as plain text
            sender = self.extract_sender(email_content)
            body = self.extract_body(email_content)
            text_for_analysis = body

        result = {
            'sender': sender,
            'urgency': self.detect_urgency(text_for_analysis).value,
            'timestamp': datetime.utcnow().isoformat(),
            'summary': body[:100] + '...' if len(body) > 100 else body,
            'body': body
        }

        if self.intent_classifier:
            intent_result = self.intent_classifier(text_for_analysis)
            # If intent_result is an Enum, get its value
            if hasattr(intent_result, "value"):
                result['intent'] = intent_result.value
            else:
                result['intent'] = intent_result
        else:
            result['intent'] = Intent.OTHER.value


        return result
# Example Usage
if __name__ == "__main__":
    def mock_classifier(email_text):
        if "invite" in email_text.lower():
            return Intent.INVITE.value
        elif "complaint" in email_text.lower():
            return Intent.COMPLAINT.value
        return Intent.OTHER.value

    agent = EmailAgent(intent_classifier=mock_classifier)

    sample_email = """From: "John Doe" <john.doe@example.com>
Subject: Invitation to Annual Conference

Dear Team,

You are cordially invited to our annual conference on September 15th.
This is an important event with all key stakeholders attending.

Please RSVP by August 30th.

Best regards,
John Doe
"""

    print("Processing email:")
    print(agent.process_email(sample_email))
