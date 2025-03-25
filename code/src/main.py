import os
import pdfplumber
import pandas as pd
from transformers import pipeline
import re
from datetime import datetime
import hashlib
from typing import Dict, List, Tuple
import nltk
nltk.download('punkt_tab')

from collections import Counter

# Initialize NLP models
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
ner_model = pipeline("ner", model="dslim/bert-base-NER")

# Predefined request types and sub-types
REQUEST_TYPES = {
    "Adjustment":[],
    "AU transfer":[],
    "Closing Notice": ["Reallocation Fees", "Amendment Fees", "Reallocation Principal"],
    "Commitment Change": ["Cashless Roll", "Decrease", "Increase"],
    "Fee Payment": ["Ongoing Fee", "Letter of Credit Fee"],
    "Money Movement-Inbound": ["Principal", "Interest", "Principal+Interest", "Principal+Interest+Fee"]
}


class EmailProcessor:
    def __init__(self, input_directory: str):
        self.input_directory = input_directory
        self.processed_emails = set()  # For duplicate detection
        self.extraction_rules = {
            "deal_name": r"(?:deal|transaction|agreement)\s*[:#-]?\s*([A-Za-z0-9_-]+)",
            "amount": r"(?:amount|sum|total)?\s*\$?([\d,]+\.?\d*)",
            "expiration_date": r"(?:expir|due|valid until|date)[\s:]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
        }

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file using pdfplumber"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                return text.strip()
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""

    def classify_request_type(self, text: str) -> Tuple[str, str]:
        """Classify email into request type and sub-request type"""
        text = text.lower()

        # Priority 1: Check email content for request type
        for req_type, sub_types in REQUEST_TYPES.items():
            if req_type.lower() in text:
                # Find specific sub-type
                for sub_type in sub_types:
                    if sub_type.lower() in text:
                        return req_type, sub_type
                return req_type, sub_types[0] if sub_types else ""


        return "Unknown", "Unknown"

    def extract_fields(self, text: str, request_type: str) -> Dict:
        """Extract configurable fields based on request type"""
        fields = {}

        for field, pattern in self.extraction_rules.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1)
                if field == "expiration_date":
                    try:
                        value = datetime.strptime(value, "%m/%d/%Y").date()
                    except:
                        try:
                            value = datetime.strptime(value, "%d/%m/%Y").date()
                        except:
                            continue
                elif field == "amount":
                    value = float(value.replace(",", ""))
                fields[field] = value

        return fields

    def detect_primary_intent(self, text: str) -> str:
        """Detect primary intent in multi-request emails"""
        sentences = nltk.sent_tokenize(text)
        intent_scores = {}

        for req_type in REQUEST_TYPES.keys():
            score = sum(1 for sentence in sentences if req_type.lower() in sentence.lower())
            intent_scores[req_type] = score

        return max(intent_scores.items(), key=lambda x: x[1])[0] if intent_scores else "Unknown"

    def is_duplicate(self, text: str) -> bool:
        """Detect duplicate emails using hash of content"""
        email_hash = hashlib.md5(text.encode()).hexdigest()
        if email_hash in self.processed_emails:
            return True
        self.processed_emails.add(email_hash)
        return False

    def process_emails(self) -> List[Dict]:
        """Main processing function"""
        results = []

        for filename in os.listdir(self.input_directory):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(self.input_directory, filename)
                 #file_path = os.path.join(self.input_directory, filename)
                # Extract text from PDF
                email_text = self.extract_text_from_pdf(file_path)
                if not email_text:
                    continue

                # Check for duplicates
                if self.is_duplicate(email_text):
                    print(f"Duplicate email detected: {filename}")
                    continue

                # Classify request type (priority to email content)
                request_type, sub_request_type = self.classify_request_type(email_text)

                # Handle multi-request emails
                if sum(1 for rt in REQUEST_TYPES.keys() if rt.lower() in email_text.lower()) > 1:
                    primary_intent = self.detect_primary_intent(email_text)
                    request_type = primary_intent

                # Extract fields
                fields = self.extract_fields(email_text, request_type)

                # Extract from attachments if needed (assuming separate PDFs)
                attachment_fields = {}
                if "amount" not in fields:  # Prioritize attachment for numeric fields
                    attachment_path = file_path.replace(".pdf", "_attachment.pdf")
                    if os.path.exists(attachment_path):
                        attachment_text = self.extract_text_from_pdf(attachment_path)
                        attachment_fields = self.extract_fields(attachment_text, request_type)
                        fields.update({k: v for k, v in attachment_fields.items() if k in ["amount"]})

                result = {
                   # "filename": filename,
                    "request_type": request_type,
                    "sub_request_type": sub_request_type,
                    "extracted_fields": fields,
                    "processed_date": datetime.now().isoformat()
                }
                print(f"File Name {filename}")
                print(result)
                results.append(result)

        return results


def main():
    # Example usage
    input_dir = r"C:\Project\files"
    processor = EmailProcessor(input_dir)
    results = processor.process_emails()

    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv("processed_emails.csv", index=False)

    # Print summary
    #print(f"Processed {len(results)} emails")
    #print("\nRequest Type Distribution:")
    #print(df["request_type"].value_counts())


if __name__ == "__main__":
    main()
