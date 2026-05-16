import torch
import datasets
import wandb

from loguru import logger
import peft

import utils
from utils_llm import DatasetRegistry
from llm.gsm8k_utils import gsm8k_exact_match
from llm.commonsenseqa_utils import COMMONSENSEQA_CHOICES

import warnings

DATASETS = ["mathqa", "coin_flip"]

MULTIPLE_CHOICE_DATASETS = {"bigbench_date", "object_tracking", "mathqa"}

warnings.filterwarnings("ignore")

from optimizers.main import get_optimizer

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    pipeline,
)


class Finetuner:
    """Main class for downstream finetuning"""

    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.builder = None
        self.setup_logging()
        if self.args.wandb:
            self.setup_wandb()

    def setup_logging(self):
        """Setup logging configuration"""
        logger.info(f"Starting downstream finetuning for dataset: {self.args.dataset}")

    def setup_wandb(self):
        """Setup Weights & Biases logging"""
        wandb.init(
            project=self.args.wandb_project,
            tags=[self.args.model, self.args.dataset, self.args.optimizer],
            name=self.args.run_name,
            config=self.args,
        )

    def load_model_and_tokenizer(self):
        """Load model and tokenizer with appropriate configurations"""
        logger.info("Loading model and tokenizer...")

        # Determine optimal settings based on GPU capability
        # [TODO] add flash_attention
        if torch.cuda.get_device_capability()[0] >= 8:
            attn_implementation = "eager"  # fix flash_attention_2
        else:
            attn_implementation = "eager"

        # Set dtype
        if self.args.dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif self.args.dtype == "float16":
            torch_dtype = torch.float16
        elif self.args.dtype == "float32":
            torch_dtype = torch.float32
        elif self.args.dtype == "float64":
            torch_dtype = torch.float64
        # Setup quantization config
        if self.args.quant_bit == 8:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch_dtype,
            )
        elif self.args.quant_bit == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
            quantization_config=bnb_config,
            attn_implementation=attn_implementation,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model,
            use_fast=self.args.use_fast_tokenizer,
            padding_side=self.args.padding_side,
        )
        self.tokenizer.pad_token_id = 0

        # Resize token embeddings if necessary
        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        if len(self.tokenizer) > embedding_size:
            self.model.resize_token_embeddings(len(self.tokenizer))

        logger.info("Model and tokenizer loaded successfully.")

    def setup_peft(self):
        """Setup PEFT (Parameter Efficient Fine-Tuning) adapters"""
        logger.info("Setting up PEFT adapters...")

        peft_args = utils.get_peft_arguments(self.args)
        if peft_args is not None:
            self.model = peft.get_peft_model(self.model, peft_args)

        # Print trainable parameters info
        tr_param_count, all_param_count, tr_persent = utils.print_trainable_params(
            self.model, verbose=True
        )

        if self.args.wandb:
            wandb.log(
                {
                    "trainable_params_count": tr_param_count,
                    "total_param_count": all_param_count,
                    "trainable_params_percentage": tr_persent,
                }
            )

    def load_datasets(self):
        """Load and prepare both training and evaluation datasets"""
        logger.info("Loading datasets...")

        # Set dataset paths and create builder
        DatasetRegistry.set_dataset_paths(self.args)
        self.builder = DatasetRegistry.create_builder(self.args)
        data = self.builder.get_data()

        # Extract train and eval data
        train_questions, train_answers = data["train"]
        eval_questions, eval_answers = data["eval"]
        train_choices = data.get("train_choices", [])
        eval_choices = data.get("eval_choices", [])

        # Create datasets
        self.train_dataset = None
        self.eval_dataset = None

        if train_questions and len(train_questions) > 0:
            train_dict = {
                "question": list(train_questions),
                "response": train_answers,
                "raw_x": train_questions,
                "raw_y": train_answers,
            }
            if train_choices:
                train_dict["raw_choices"] = train_choices
            self.train_dataset = datasets.Dataset.from_dict(train_dict)

        if eval_questions and len(eval_questions) > 0:
            eval_dict = {
                "question": list(eval_questions),
                "response": eval_answers,
                "raw_x": eval_questions,
                "raw_y": eval_answers,
            }
            if eval_choices:
                eval_dict["raw_choices"] = eval_choices
            self.eval_dataset = datasets.Dataset.from_dict(eval_dict)

    def prepare_training_dataset(self):
        """Prepare dataset for training"""
        if self.train_dataset is None:
            logger.warning("No training dataset available")
            return None

        logger.info("Preparing training dataset...")
        return self.builder.preprocess_dataset(
            self.tokenizer,
            self.args.max_seq_length,
            self.args.seed,
            self.train_dataset,
            eval_mode=False,
        )

    def prepare_evaluation_dataset(self):
        """Prepare dataset for evaluation"""
        if self.eval_dataset is None:
            logger.warning("No evaluation dataset available")
            return None

        logger.info("Preparing evaluation dataset...")
        dataset = self.eval_dataset.map(
            lambda sample: self.builder.create_prompt_formats(sample, eval_mode=True)
        )
        return dataset

    def get_optimizer(self):
        """Get optimizer for training"""
        optimizer = get_optimizer(self.args, self.model)
        return optimizer

    def train(self):
        """Execute training process"""
        if self.args.do_not_train:
            logger.info("Training skipped (do_not_train=True)")
            return

        logger.info(f"Starting training for dataset: {self.args.dataset}")

        # Prepare training dataset
        train_dataset = self.prepare_training_dataset()
        if train_dataset is None:
            logger.error("No training data available")
            return

        # Setup trainer
        training_args = TrainingArguments(
            do_train=not self.args.do_not_train,
            do_eval=not self.args.do_not_eval,
            do_predict=self.args.do_predict,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=(
                self.args.eval_batch_size
                if self.args.eval_batch_size
                else self.args.batch_size
            ),
            gradient_accumulation_steps=self.args.grad_acc_steps,
            lr_scheduler_type=self.args.lr_scheduler_type,
            warmup_steps=self.args.warmup_steps,
            learning_rate=self.args.lr,
            num_train_epochs=self.args.n_epoches_train,
            max_steps=self.args.max_steps_train,
            logging_steps=self.args.logging_steps,
            eval_strategy="steps" if args.eval_steps else args.eval_strategy,
            eval_steps=self.args.eval_steps,
            save_strategy=self.args.save_strategy,
            save_steps=self.args.save_steps,
            bf16=(self.args.dtype == "bfloat16"),
            fp16=(self.args.dtype == "float16"),
            logging_dir=f"./src/fine_tuning/llm/{self.args.results_path}/{self.args.run_name}",
            run_name=self.args.run_name,
            report_to=["wandb"] if self.args.wandb else ["none"],
        )

        optimizer = self.get_optimizer()

        trainer = Trainer(
            model=self.model,
            train_dataset=train_dataset,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            optimizers=[optimizer, None],  # Scheduler will be added in the hf trainer
        )

        # Clean up memory before training
        import gc

        gc.collect()
        torch.cuda.empty_cache()

        # Train the model
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        logger.info(f"Training completed. Metrics: {metrics}")

    def evaluate(self):
        """Execute evaluation process"""
        if self.args.do_not_eval:
            logger.info("Evaluation skipped (do_not_eval=True)")
            return 0, 0

        logger.info(f"Starting evaluation for dataset: {self.args.dataset}")

        # Prepare evaluation dataset
        eval_dataset = self.prepare_evaluation_dataset()
        if eval_dataset is None:
            logger.error("No evaluation data available")
            return 0, 0

        # Setup text generation pipeline
        generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
        )

        # Evaluate model
        correct, total = 0, 0

        logger.info(f"Evaluation sample prompt: {eval_dataset['text'][0]}")

        for i, item in enumerate(eval_dataset):
            try:
                predicted_response, is_correct = self.score_eval_item(item, generator)

                if is_correct:
                    correct += 1

                # Debug output
                print(f">>Prediction<<: {predicted_response}")
                print(f">>>>Answer<<<<: {item['raw_y']}")

                total += 1

                # Progress update
                accuracy = (correct / total) * 100 if total > 0 else 0
                print(f"[{i+1}/{len(eval_dataset)}] Accuracy: {accuracy:.2f}%")
                print("=" * 50)

            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                continue

        return correct, total

    def get_eval_max_new_tokens(self, item):
        if self.args.dataset == "gsm8k":
            return 64
        return len(item["raw_y"]) + 2

    def score_eval_item(self, item, generator):
        if self.args.dataset in MULTIPLE_CHOICE_DATASETS:
            predicted = self.score_multiple_choice(
                item["text"],
                item.get("raw_choices", []),
                normalize=True,
            )
            return predicted, predicted == item["raw_y"]

        predicted = generator(
            item["text"],
            max_new_tokens=self.get_eval_max_new_tokens(item),
            num_return_sequences=1,
            do_sample=False,
        )[0]["generated_text"]

        if self.args.dataset == "gsm8k":
            return predicted, gsm8k_exact_match(predicted, item["raw_y"])

        pred_norm = predicted.replace(" ", "").replace("\n", "")
        gold_norm = item["raw_y"].replace(" ", "").replace("\n", "")
        return predicted, gold_norm in pred_norm

    def score_multiple_choice(self, prompt, choices, normalize=False):
        best_choice = None
        best_score = None

        prompt_tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=min(self.args.max_seq_length, self.tokenizer.model_max_length),
            add_special_tokens=False,
        )
        prompt_len = prompt_tokens["input_ids"].shape[1]

        for choice in choices:
            full_tokens = self.tokenizer(
                f"{prompt} {choice}",
                return_tensors="pt",
                truncation=True,
                max_length=min(self.args.max_seq_length, self.tokenizer.model_max_length),
                add_special_tokens=False,
            )

            full_input_ids = full_tokens["input_ids"].to(self.model.device)
            full_attention_mask = full_tokens["attention_mask"].to(self.model.device)
            continuation_len = full_input_ids.shape[1] - prompt_len
            if continuation_len <= 0:
                continue

            labels = full_input_ids.clone()
            labels[:, :prompt_len] = -100

            with torch.no_grad():
                outputs = self.model(
                    input_ids=full_input_ids,
                    attention_mask=full_attention_mask,
                    labels=labels,
                )

            score = -outputs.loss.item() if normalize else -outputs.loss.item() * continuation_len
            if best_score is None or score > best_score:
                best_score = score
                best_choice = choice

        return best_choice or ""

    def log_final_results(self, correct, total):
        """Log final evaluation results"""
        if total > 0:
            final_accuracy = (correct / total) * 100
            logger.info(f"[FINAL] Accuracy: {final_accuracy:.2f}%")

            if self.args.wandb:
                wandb.log({"final_accuracy": final_accuracy})
        else:
            logger.info("No samples were successfully evaluated.")

    def run(self):
        """Main execution flow"""
        logger.info("Starting finetuning pipeline")

        utils.set_global_seed(self.args.seed)

        # Load model and setup PEFT
        self.load_model_and_tokenizer()
        self.setup_peft()

        # Load datasets
        self.load_datasets()

        # Execute training and evaluation
        self.train()
        correct, total = self.evaluate()

        # Log results
        self.log_final_results(correct, total)
        logger.info("Pipeline completed successfully")


def main(args):
    """Main entry point"""
    # Create and run finetuner
    finetuner = Finetuner(args)
    finetuner.run()


if __name__ == "__main__":
    main(None)
