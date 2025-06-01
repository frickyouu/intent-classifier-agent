import pandas as pd
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset, ClassLabel
import logging
import os
import torch

# ------------------- Key Change 1: Force CUDA check and exit if unavailable -------------------
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not detected! Install CUDA-enabled PyTorch first.")
print(f"Using device: {torch.cuda.get_device_name(0)}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FormatTrainer:
    def __init__(self, csv_path="C:\\Users\\VANSH\\OneDrive\\Documents\\python projects data\\ai agent classifier\\synthetic_format_dataset.csv"):
        self.csv_path = csv_path
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.class_names = ["PDF", "EMAIL", "JSON", "UNKNOWN"]
        self.class_label = ClassLabel(names=self.class_names)
        self.device = torch.device("cuda")  # Key Change 2: Explicit device assignment

    def load_dataset(self):
        df = pd.read_csv(self.csv_path)
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(
            lambda x: {"label": self.class_label.str2int(x["label"])},
            batched=False
        )
        return dataset

    def train(self, output_dir="./models/format_model", epochs=3):
        dataset = self.load_dataset()

        def tokenize(batch):
            return self.tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=512
            )

        dataset = dataset.map(tokenize, batched=True)

        # Key Change 3: Move model to GPU immediately
        model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=len(self.class_names),
            problem_type="single_label_classification"
        ).to(self.device)  # <-- Force model to GPU here

        # Key Change 4: Optimize TrainingArguments for GPU
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=16,  # Increased batch size for GPU efficiency
            num_train_epochs=epochs,
            logging_dir=os.path.join(output_dir, "logs"),
            save_strategy="epoch",
            report_to="none",
            fp16=True,  # Enable mixed-precision training (uses GPU tensor cores)
            gradient_accumulation_steps=2,  # Better GPU utilization
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )

        logger.info("Starting training...")
        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)  # <-- ADD THIS LINE
        # Save label mapping
        with open(os.path.join(output_dir, "label_mapping.txt"), "w") as f:
            for label in self.class_names:
                f.write(label + "\n")
        logger.info(f"Model and tokenizer saved to {output_dir}")

        # Save label mapping
        with open(os.path.join(output_dir, "label_mapping.txt"), "w") as f:
            for label in self.class_names:
                f.write(label + "\n")
        logger.info(f"Model saved to {output_dir}")

if __name__ == "__main__":
    trainer = FormatTrainer()
    trainer.train(epochs=3)
