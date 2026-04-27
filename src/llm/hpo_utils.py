import os


def get_optuna_metric_name(dataset_name: str) -> str:
    dataset_name = dataset_name.lower()
    if dataset_name in [
        "aqua",
        "gsm8k",
        "commonsensqa",
        "boolq",
        "addsub",
        "multiarith",
        "singleeq",
        "strategyqa",
        "svamp",
        "bigbench_date",
        "object_tracking",
        "coin_flip",
        "last_letters",
        "math_qa",
        "mathqa",
        "hella_swag",
        "arc_challenge",
    ]:
        return "eval_accuracy"
    if dataset_name == "cola":
        return "eval_matthews_correlation" #MCC
    if dataset_name in ["sst2", "mnli", "qnli", "rte", "wnli"]:
        return "eval_accuracy"
    if dataset_name in ["mrpc", "qqp"]:
        return "eval_f1"
    if dataset_name == "stsb":
        return "eval_pearson"
    if dataset_name in ["squad", "squad_v2"]:
        return "eval_f1"

    raise ValueError(f"Unsupported dataset for HPO: {dataset_name}")


def suggest_hparams(trial, args):
    """
    Подбираем только lr и lora_r.
    Остальное фиксируется через args.
    """
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    if getattr(args, "hpo_lr_only", False):
        return {"lr": lr}
    lora_r = trial.suggest_categorical("lora_r", [4, 8])
    return {"lr": lr, "lora_r": lora_r}


def build_study_name(args) -> str:
    model_name = args.model.replace("/", "_")
    return f"{model_name}__{args.dataset}__{args.ft_strategy}"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
