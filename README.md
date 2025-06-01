# intent-classifier-agent #
## ClassifierAgent##
This README provides setup and usage instructions for classifier1.py, a Python script that automatically classifies the format and intent of input files (PDF, Email, or JSON), and routes them to specialized agents for further processing.

## Features ##

Detects input file format: PDF, Email (.eml), or JSON.

Classifies the intent of the document (e.g., Invoice, Complaint, Request for Quotation, Regulation, Other).

Extracts and logs text content from the file.

Routes JSON and Email files to specialized agents for further processing.

Maintains persistent logs via a shared memory database.

## Requirements ##
Python 3.8+

PyTorch

transformers

PyPDF2

joblib

Required model files and label mappings in the specified directories

Custom modules: json_agent.py, email_agent.py, shared_memory.py

## DIRECTORY STRUCTURE ##
project/
│
├── classifier1.py
├── email_agent.py
├── json_agent.py
├── shared_memory.py
├── models/
│   └── format_model/
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── label_mapping.txt
│       └── tokenizer files
├── email_intent_bert/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── label_classes.pkl
│   └── tokenizer files
└── persistent.db

