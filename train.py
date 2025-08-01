from peft import LoraConfig, PeftMixedModel, PeftModel, get_peft_model, TaskType
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.training_args import TrainingArguments
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer import Trainer
from models import DialogMessage
from datasets import Dataset
import torch
import os


class LoRATrainer:
    def __init__(self, model_name: str, output_dir: str):
        self.model_name: str = model_name
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.output_dir: str = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def prepare_dataset(self, messages: list[DialogMessage]) -> Dataset:
        """
        Prepare dataset from speaker messages for training

        Args:
            messages: List of message dictionaries

        Returns:
            Dataset for training
        """
        # Format messages
        # For Llama:
        #   "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
        text_format: str = "User: {}\n"
        formatted_texts: list[str] = [
            text_format.format(message.content) for message in messages
        ]
        full_text: str = "".join(formatted_texts)

        # Tokenize all texts
        tokenized_data: BatchEncoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        return Dataset.from_dict(
            {
                "input_ids": [tokenized_data["input_ids"].squeeze().tolist()],
                "attention_mask": [
                    tokenized_data["attention_mask"].squeeze().tolist()
                ],
            }
        )

    def train_model(self, speaker: str, messages: list[DialogMessage]) -> str:
        """
        Train the model using LoRA for the specified speaker

        Args:
            speaker: Speaker name for training
            messages: List of message dictionaries

        Returns:
            Path to the trained model
        """
        # Initializations
        quantization_config: BitsAndBytesConfig = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

        model_instance: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
        )

        lora_config: LoraConfig = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            target_modules=["c_attn", "c_proj"],
            lora_dropout=0.1,
            bias="none",
        )

        model: PeftModel | PeftMixedModel = get_peft_model(
            model_instance, lora_config
        )

        dataset: Dataset = self.prepare_dataset(messages)

        training_args: TrainingArguments = TrainingArguments(
            output_dir=os.path.join(self.output_dir, speaker),
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=10,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_steps=100,
            save_strategy="steps",
            remove_unused_columns=False,
        )

        data_collator: DataCollatorForLanguageModeling = (
            DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        )

        trainer: Trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            processing_class=self.tokenizer,
        )

        # Train the model
        trainer.train()

        # Save the model
        path: str = os.path.join(self.output_dir, speaker)
        trainer.save_model(path)
        self.tokenizer.save_pretrained(path)
        return path
