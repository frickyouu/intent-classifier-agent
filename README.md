# intent-classifier-agent #
## ClassifierAgent ##
This README provides setup and usage instructions for classifier1.py, a Python script that automatically classifies the format and intent of input files (PDF, Email, or JSON), and routes them to specialized agents for further processing.

## Features ##

Detects input file format: PDF, Email (.eml), or JSON.

Classifies the intent of the document (e.g., Invoice, Complaint, Request for Quotation, Regulation, Other).

Extracts and logs text content from the file.

Routes JSON and Email files to specialized agents for further processing.

Maintains persistent logs via a shared memory database.
#### dependencies ###
```pip install torch transformers PyPDF2 joblib```


## Requirements ##
Python 3.8+

PyTorch

transformers

PyPDF2

joblib

Required model files and label mappings in the specified directories

Custom modules: json_agent.py, email_agent.py, shared_memory.py

## DIRECTORY STRUCTURE ##
```
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
```

## Usage
1) Prepare your models and label files

Place your format classification model and label mapping in ./models/format_model/.

Place your intent classification model and label_classes.pkl in ./email_intent_bert/.

2) Run the script

bash
```python classifier1.py ```
3) Follow the prompts

When prompted, enter the path to a PDF, JSON, or Email (.eml) file.

The script will classify the file format and intent, process the file, and display the results.

You can process multiple files in sequence.

## Customization
Adding new intents: Update your intent model and label_classes.pkl.

Extending agents: Modify or extend json_agent.py and email_agent.py for custom logic.

Persistent logging: All processed results are logged in persistent.db via SharedMemory.
