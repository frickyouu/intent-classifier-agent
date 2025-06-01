from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from enum import Enum
import logging
import json
import email
from PyPDF2 import PdfReader
import re
import io
import os
import joblib
from json_agent import JSONAgent  # NEW
from email_agent import EmailAgent  # NEW
from shared_memory import SharedMemory
from email import policy
from email.parser import BytesParser

class FileFormat(Enum):
    PDF = "PDF"
    EMAIL = "EMAIL"
    JSON = "JSON"
    UNKNOWN = "UNKNOWN"

class Intent(Enum):
    INVOICE = "Invoice"
    RFQ = "Request for Quotation"
    COMPLAINT = "Complaint"
    REGULATION = "Regulation"
    OTHER = "Other"

class ClassifierAgent:
    def __init__(self):
        
        # Initialize format classification model
        # In __init__:
        self.format_model_path = "./models/format_model"
        self.format_tokenizer = AutoTokenizer.from_pretrained(self.format_model_path)
        self.format_model = AutoModelForSequenceClassification.from_pretrained(self.format_model_path)

        # Load label mapping
        with open(f"{self.format_model_path}/label_mapping.txt") as f:
            labels = [line.strip() for line in f.readlines()]
        self.format_labels = [FileFormat[label] for label in labels]

        
        # Initialize intent classification model
        self.intent_model_path = "./email_intent_bert"
        self.intent_tokenizer = AutoTokenizer.from_pretrained(self.intent_model_path)
        self.intent_model = AutoModelForSequenceClassification.from_pretrained(self.intent_model_path)

        # Load label classes (e.g., ['Invoice', 'Complaint', ...])
        self.intent_label_classes = joblib.load(f"{self.intent_model_path}/label_classes.pkl")
        
        # Configure logging
        self.logger = logging.getLogger("ClassifierAgent")
        logging.basicConfig(level=logging.INFO)
        # Initialize JSON agent
        self.json_agent = JSONAgent()  # NEW
        self.email_agent = EmailAgent(intent_classifier=self.classify_intent)  # NEW
        self.memory = SharedMemory(db_path="persistent.db")

    def classify_format(self, input_data):
        """Classify the format of the input data"""
        try:
            # First try heuristic detection
            if self._is_pdf(input_data):
                return FileFormat.PDF
            elif self._is_email(input_data):
                return FileFormat.EMAIL
            elif self._is_json(input_data):
                return FileFormat.JSON
            
            # Fall back to ML model if heuristics fail
            inputs = self.format_tokenizer(input_data[:512], return_tensors="pt", truncation=True)
            outputs = self.format_model(**inputs)
            predicted_idx = torch.argmax(outputs.logits, dim=1).item()
            return self.format_labels[predicted_idx]
            
        except Exception as e:
            self.logger.error(f"Error classifying format: {str(e)}")
            return FileFormat.UNKNOWN
    
    def classify_intent(self, text_content):
        
        """Classify the intent of the content"""
        # Keyword check for RFQ emails
        if "request for quotation" in text_content.lower() or "quotation" in text_content.lower():
            return Intent.RFQ  # [7][8]

        # Original model-based classification
        try:
            inputs = self.intent_tokenizer(text_content[:512], return_tensors="pt", truncation=True)
            outputs = self.intent_model(**inputs)
            predicted_idx = torch.argmax(outputs.logits, dim=1).item()
            predicted_label = self.intent_label_classes[predicted_idx]
            try:
                return Intent[predicted_label.upper().replace(" ", "_")]
            except KeyError:
                return Intent.OTHER
        except Exception as e:
            self.logger.error(f"Error classifying intent: {str(e)}")
            return Intent.OTHER

    
    def process_input(self, input_data):
        """Main processing method"""
        # Determine format
        format_type = self.classify_format(input_data)
        
        # Extract text content based on format
        text_content = self._extract_text(input_data, format_type)
        
        # Determine intent
        intent = self.classify_intent(text_content)
        
       # Log results
        self._log_classification(format_type, intent)

        # Basic metadata (can expand later)
        metadata = {
            "format": format_type.value,
            "intent": intent.value,
            "text_snippet": text_content[:200]
        }

        # Route to agent
        # Route to agent
        if format_type == FileFormat.JSON:
            agent_result = self.json_agent.process(input_data, metadata)
            self.memory.log_json(source_id="json_input", json_result=agent_result)
            return {
                "format": format_type.value,
                "intent": intent.value,
                "agent_result": agent_result
            }

        elif format_type == FileFormat.EMAIL:
            agent_result = self.email_agent.process_email(input_data)
                # Ensure intent is a string for logging
            if "intent" in agent_result and hasattr(agent_result["intent"], "value"):
                agent_result["intent"] = agent_result["intent"].value
            self.memory.log_email(source_id=agent_result.get("sender", "unknown_email"), email_result=agent_result)
            return {
                "format": format_type.value,
                "intent": intent.value,
                "agent_result": agent_result
            }

        # Default return if no agent matched
        return {
            "format": format_type.value,
            "intent": intent.value,
            "text_content": text_content[:200]
        }


    
    def _is_pdf(self, data):

        try:
            if isinstance(data, bytes):
                PdfReader(io.BytesIO(data))
            else:
                with open(data, "rb") as f:
                    PdfReader(f)
            return True
        except:
            return False
    
    def _is_email(self, data):
        try:
            if isinstance(data, str):
                headers = ['From:', 'To:', 'Subject:']
                found = sum(1 for h in headers if re.search(rf'^{h}', data, re.MULTILINE | re.IGNORECASE))
                if found >= 2:
                    return True
        # Fallback to original logic
            if isinstance(data, str) and os.path.isfile(data):
                with open(data, "rb") as f:
                    data = f.read()
            if isinstance(data, bytes):
                msg = BytesParser(policy=policy.default).parsebytes(data)
            else:
                msg = email.message_from_string(data)
            return msg.get('Subject') is not None and msg.get('From') is not None
        except Exception:
            return False



    
    def _is_json(self, data):
        """Check if data is JSON"""
        try:
            json.loads(data)
            return True
        except:
            return False
    
    def _extract_text(self, data, format_type):
        try:
            if format_type == FileFormat.PDF:
                if isinstance(data, bytes):
                    reader = PdfReader(io.BytesIO(data))
                else:
                    with open(data, "rb") as f:
                        reader = PdfReader(f)
                return " ".join([page.extract_text() for page in reader.pages])
            elif format_type == FileFormat.EMAIL:
                if isinstance(data, str) and os.path.isfile(data):
                    with open(data, "rb") as f:
                        data = f.read()
                            
                if isinstance(data, bytes):
                    msg = BytesParser(policy=policy.default).parsebytes(data)
                else:
                    msg = email.message_from_string(data)
                # Extract plain text body
                if msg.is_multipart():
                    parts = [part for part in msg.walk() if part.get_content_type() == 'text/plain']
                    return ' '.join(
                        part.get_content().strip() if hasattr(part, "get_content") else part.get_payload(decode=True).decode(errors='ignore')
                        for part in parts
                    )
                else:
                    if hasattr(msg, "get_content"):
                        return msg.get_content().strip()
                    else:
                        return msg.get_payload(decode=True).decode(errors='ignore')
            elif format_type == FileFormat.JSON:
                json_data = json.loads(data)
                return json.dumps(json_data)
            else:
                return str(data)
        except Exception as e:
            self.logger.error(f"Error extracting text: {str(e)}")
            return ""
    
    def _log_classification(self, format_type, intent):
        """Log classification results"""
        self.logger.info(f"Classified: Format={format_type.value}, Intent={intent.value}")


def handle_file_upload():
    file_path = input("Enter the path to your PDF, JSON, or email file: ").strip('"')
    if not os.path.exists(file_path):
        print("Error: File not found!")
        return None, None

    # Detect file type by extension
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        with open(file_path, 'rb') as f:
            return f.read(), 'PDF'
    elif ext == '.eml':
        with open(file_path, 'rb') as f:
            return f.read(), 'EMAIL'
    elif ext == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read(), 'JSON'
    else:
        print("Unsupported file type. Please provide a PDF, .eml, or .json file.")
        return None, None

if __name__ == "__main__":
    classifier = ClassifierAgent()
    while True:
        input_data, file_type = handle_file_upload()
        if input_data:
            result = classifier.process_input(input_data)
            print("\nClassification Result:")
            print(result)
        again = input("\nProcess another file? (y/n): ").strip().lower()
        if again != 'y':
            print("Exiting...")
            break
