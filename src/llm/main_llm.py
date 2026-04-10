import os

from loguru import logger
import peft

import utils

import datasets
import torch

import transformers
from transformers import (
    set_seed,
)

# FIX: decide what to do with NLG and remove (probably)
from transformers.utils import is_offline_mode
from filelock import FileLock
import nltk

from models_llm import create_model_framework
from problems_llm import (
    CAUSAL_LM_DATASETS, GLUE_DATASETS, SQUAD_DATASETS, NLG_DATASETS,
    create_problem,
)
from optimizers.main import get_optimizer

DATASETS = CAUSAL_LM_DATASETS + GLUE_DATASETS + SQUAD_DATASETS + NLG_DATASETS

def get_peft_config(args, target_modules):
    if args.ft_strategy == "LoRA":
        peft_config = peft.LoraConfig(
            task_type=args.task_type,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
        )
    elif args.ft_strategy == "LoKR":
        peft_config = peft.LoKrConfig(
            task_type=args.task_type,
            r=args.lora_r,
            alpha=args.lora_alpha,
            rank_dropout=args.lora_dropout,
            target_modules=target_modules,
        )
    elif args.ft_strategy == "LoHA":
        peft_config = peft.LoHaConfig(
            task_type=args.task_type,
            r=args.lora_r,
            alpha=args.lora_alpha,
            rank_dropout=args.lora_dropout,
            target_modules=target_modules,
        )
    elif args.ft_strategy == "VERA":
        peft_config = peft.VeraConfig(
            task_type=args.task_type,
            r=args.lora_r,
            vera_dropout=args.lora_dropout,
            target_modules=target_modules,
        )
    elif args.ft_strategy == "ADALoRA":
        peft_config = peft.AdaLoraConfig(
            task_type=args.task_type,
            target_r=args.lora_r,
            target_modules=target_modules,
            total_step = args.max_steps_train,
        )
    elif args.ft_strategy == "DoRA":
        peft_config = peft.LoraConfig(
            task_type=args.task_type,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            use_dora=True,
            target_modules=target_modules,
        )
    elif args.ft_strategy == "rsLoRA":
        peft_config = peft.LoraConfig(
            task_type=args.task_type,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            use_rslora=True,
            target_modules=target_modules,
        )
    elif args.ft_strategy == "WeightLoRA":
        peft_config = peft.LoraConfig(
            task_type=args.task_type,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            use_weight_lora=True,
            target_modules=target_modules,
        )
    elif args.ft_strategy == "GraLoRA":
        peft_config = peft.GraloraConfig(
            task_type=args.task_type,
            r=args.lora_r,
            alpha=args.lora_alpha,
            gralora_dropout=args.lora_dropout,
            target_modules=target_modules,
        )
    elif args.ft_strategy == "Full":
        peft_config = None
    else:
        #~ fix error message
        raise ValueError(f"Unknown ft_strategy={args.ft_strategy}, choose from availabale opions:")

    return peft_config


def run_single_experiment(args):
    set_seed(args.seed)

    args.dataset = args.dataset.lower()
    if args.dataset in CAUSAL_LM_DATASETS:
        args.task_type = "CAUSAL_LM"
    elif args.dataset in GLUE_DATASETS:
        args.task_type = "SEQ_CLS"
    elif args.dataset in SQUAD_DATASETS:
        args.task_type = "QUESTION_ANS"
    elif args.dataset in NLG_DATASETS:
        args.task_type = "SEQ_2_SEQ_LM"
    else:
        raise ValueError(f"Unknown dataset={args.dataset}, choose from available options:")

    args.do_train = not args.do_not_train
    args.do_eval = not args.do_not_eval
    args.bf16 = (args.dtype == "bfloat16")
    args.fp16 = (args.dtype == "float16")

    if args.task_type == "SEQ_2_SEQ_LM":
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except (LookupError, OSError):
            if is_offline_mode():
                raise LookupError(
                    "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
                )
            with FileLock(".lock") as lock:
                nltk.download("punkt_tab", quiet=True)

    framework = create_model_framework(args)
    model, tokenizer = framework.load_model_and_tokenizer()

    problem = create_problem(args)
    train_dataset, eval_dataset, test_dataset = problem.process_dataset(tokenizer)
    data_collator = problem.get_data_collator(model, tokenizer)
    compute_metrics = problem.get_metrics_function(tokenizer)

    peft_config = get_peft_config(args, framework.get_target_modules())

    if peft_config is not None:
        model = peft.get_peft_model(model, peft_config)

    utils.print_trainable_params(model)

    AutoTrainer = problem.get_trainer_class()
    training_args = problem.get_training_args()
    optimizer = get_optimizer(args, model)

    trainer = AutoTrainer(
        model=model,
        args=training_args,
        train_dataset=(train_dataset if args.do_train else None),
        eval_dataset=(eval_dataset if args.do_eval else None),
        optimizers=[optimizer, None],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_metrics = {}
    eval_metrics = {}
    predict_metrics = {}

    if args.do_train:
        print("~~~~~~~~~~~~~~~ TRAINING ~~~~~~~~~~~~~~~")
        results = trainer.train()
        train_metrics = results.metrics
        train_metrics["train_samples"] = (
            min(args.max_train_samples, len(train_dataset))
            if args.max_train_samples is not None
            else len(train_dataset)
        )
        train_metrics["train_memory_gb"] = torch.cuda.max_memory_allocated() / 2**30
        trainer.log_metrics("train", train_metrics)

    if args.do_eval:
        print("~~~~~~~~~~~~~~~ VALIDATING ~~~~~~~~~~~~~~~")

        max_eval_samples = (
            min(args.max_eval_samples, len(eval_dataset))
            if args.max_eval_samples is not None
            else len(eval_dataset)
        )

        if args.task_type == "QUESTION_ANS":
            eval_samples = problem.eval_dataset
            if len(eval_samples) > max_eval_samples:
                eval_samples = eval_samples.select(range(max_eval_samples))
            eval_kwargs = {"eval_samples": eval_samples}
        elif args.task_type == "SEQ_2_SEQ_LM":
            eval_kwargs = {
                "max_length": (
                    training_args.generation_max_length
                    if training_args.generation_max_length is not None
                    else args.val_max_target_length
                ),
                "num_beams": args.num_beams
            }
        else:
            eval_kwargs = {}

        eval_metrics = trainer.evaluate(
            eval_dataset,
            metric_key_prefix="eval",
            **eval_kwargs
        )
        eval_metrics["eval_samples"] = max_eval_samples
        trainer.log_metrics("eval", eval_metrics)

        #if args.task_type == "SEQ_CLS" and args.dataset == "mnli":
        #    raise NotImplementedError("Have not added mismatched evaluation yet")

    if args.do_predict:
        print("~~~~~~~~~~~~~~~ PREDICTING ~~~~~~~~~~~~~~~")

        max_test_samples = (
            min(args.max_test_samples, len(test_dataset))
            if args.max_test_samples is not None
            else len(test_dataset)
        )

        if args.task_type == "QUESTION_ANS":
            test_samples = problem.test_dataset
            if len(test_samples) > max_test_samples:
                test_samples = test_samples.select(range(max_test_samples))
            predict_kwargs = {"test_samples": test_samples}
        elif args.task_type == "SEQ_2_SEQ_LM":
            predict_kwargs = {
                "max_length": (
                    training_args.generation_max_length
                    if training_args.generation_max_length is not None
                    else args.val_max_target_length
                ),
                "num_beams": args.num_beams
            }
        else:
            predict_kwargs = {}

        results = trainer.predict(
            test_dataset,
            metric_key_prefix="predict",
            **predict_kwargs
        )

        predict_metrics = results.metrics
        predict_metrics["predict_samples"] = max_test_samples
        trainer.log_metrics("predict", predict_metrics)

        if args.task_type == "SEQ_CLS" and args.dataset == "mnli":
            raise NotImplementedError("Have not added mismatched prediction yet")

        if trainer.is_world_process_zero() and args.task_type == "SEQ_2_SEQ_LM" and args.predict_with_generate:
            preds = tokenizer.batch_decode(
                results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            preds = [p.strip() for p in preds]
            output_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
            with open(output_file, 'w') as output:
                output.write("\n".join(preds))

    # Освобождение памяти между trial
    del trainer, model, optimizer
    torch.cuda.empty_cache()

    return {
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "predict_metrics": predict_metrics,
    }

def main(args):
    return run_single_experiment(args)


if __name__ == '__main__':
    main(None)