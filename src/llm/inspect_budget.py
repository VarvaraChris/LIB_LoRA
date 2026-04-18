import copy
import gc
import sys
import warnings

import peft
import torch
from transformers import logging as transformers_logging

sys.path.extend(["src/libsvm", "src/cv", "src/llm"])

from config import parse_args
import utils
from llm.main_llm import DATASETS, get_peft_config, resolve_target_modules
from llm.models_llm import create_model_framework
from llm.problems_llm import (
    CAUSAL_LM_DATASETS,
    GLUE_DATASETS,
    NLG_DATASETS,
    SQUAD_DATASETS,
)


TASK_TYPES = {
    "CAUSAL_LM": set(CAUSAL_LM_DATASETS),
    "SEQ_CLS": set(GLUE_DATASETS),
    "QUESTION_ANS": set(SQUAD_DATASETS),
    "SEQ_2_SEQ_LM": set(NLG_DATASETS),
}


def get_cli_value(flag):
    if flag not in sys.argv:
        return None
    i = sys.argv.index(flag)
    return sys.argv[i + 1] if i + 1 < len(sys.argv) else None


def parse_list(raw, cast=str):
    if not raw:
        return []
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def strip_custom_args():
    custom_flags = {"--ft_strategies", "--r_values", "--quiet"}
    cleaned = [sys.argv[0]]
    skip_next = False

    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in custom_flags:
            if arg != "--quiet":
                skip_next = True
            continue
        cleaned.append(arg)

    sys.argv = cleaned


def infer_task_type(dataset_name):
    dataset_name = dataset_name.lower()
    for task_type, datasets in TASK_TYPES.items():
        if dataset_name in datasets:
            return task_type
    raise ValueError(
        f"Unknown dataset={dataset_name}, choose from available options: {DATASETS}"
    )


def inspect_one(base_args, ft_strategy, rank):
    args = copy.deepcopy(base_args)
    args.dataset = args.dataset.lower()
    args.task_type = infer_task_type(args.dataset)
    args.ft_strategy = ft_strategy
    args.lora_r = rank

    framework = create_model_framework(args)
    model, _ = framework.load_model_and_tokenizer()

    target_modules = resolve_target_modules(args, framework)
    peft_config = get_peft_config(args, target_modules)
    if peft_config is not None:
        model = peft.get_peft_model(model, peft_config)

    all_params, trainable_params, trainable_pct = utils.print_trainable_params(
        model, verbose=False
    )
    num_adapters = utils.count_atapters(model, ft_strategy)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return (
        ft_strategy,
        rank,
        trainable_params / 1_000_000,
        all_params / 1_000_000_000,
        trainable_pct,
        num_adapters,
        ",".join(target_modules),
    )


def main():
    ft_strategies = parse_list(get_cli_value("--ft_strategies"))
    r_values = parse_list(get_cli_value("--r_values"), int)
    quiet = "--quiet" in sys.argv

    if quiet:
        transformers_logging.set_verbosity_error()
        warnings.filterwarnings("ignore")

    strip_custom_args()
    args, _ = parse_args()

    ft_strategies = ft_strategies or [args.ft_strategy]
    r_values = r_values or [args.lora_r]

    print(
        "ft_strategy,lora_r,trainable_params_m,all_params_b,"
        "trainable_pct,num_adapters,target_modules"
    )

    for ft_strategy in ft_strategies:
        for rank in r_values:
            result = inspect_one(args, ft_strategy, rank)
            print(
                f"{result[0]},{result[1]},{result[2]:.4f},{result[3]:.4f},"
                f"{result[4]:.4f},{result[5]},\"{result[6]}\""
            )


if __name__ == "__main__":
    main()
