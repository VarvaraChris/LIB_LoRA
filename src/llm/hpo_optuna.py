import copy
import gc
import json
import os

import optuna
import torch
from loguru import logger

from llm.hpo_utils import (
    get_optuna_metric_name,
    suggest_hparams,
    build_study_name,
    ensure_dir,
)
from llm.main_llm import run_single_experiment


def objective_factory(base_args):
    metric_name = get_optuna_metric_name(base_args.dataset)

    def objective(trial):
        args = copy.deepcopy(base_args)

        #Предлагаем новые гиперпараметры
        sampled = suggest_hparams(trial, args)
        args.lr = sampled["lr"]
        args.lora_r = sampled["lora_r"]

        logger.info(
            f"[Trial {trial.number}] dataset={args.dataset}, "
            f"model={args.model}, method={args.ft_strategy}, "
            f"lr={args.lr}, r={args.lora_r}"
        )

        #Запускаем один полный эксперимент
        result = run_single_experiment(args)

        eval_metrics = result["eval_metrics"]
        if metric_name not in eval_metrics:
            raise ValueError(
                f"Metric {metric_name} not found. "
                f"Available metrics: {list(eval_metrics.keys())}"
            )

        score = eval_metrics[metric_name]

        #Сохраняем служебную информацию в trial
        trial.set_user_attr("eval_metrics", eval_metrics)
        trial.set_user_attr("lr", args.lr)
        trial.set_user_attr("lora_r", args.lora_r)

        #Освобождаем память
        gc.collect()
        torch.cuda.empty_cache()

        return score

    return objective


def run_optuna_hpo(args):
    output_dir = getattr(args, "hpo_storage_dir", "./optuna_results")
    ensure_dir(output_dir)

    study_name = build_study_name(args)
    db_path = os.path.join(output_dir, f"{study_name}_optuna_trials.db")
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )

    objective = objective_factory(args)

    study.optimize(
        objective,
        n_trials=args.hpo_n_trials,
        gc_after_trial=True,
    )

    best_summary = {
        "study_name": study_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
        "best_trial_user_attrs": study.best_trial.user_attrs,
    }

    best_path = os.path.join(output_dir, f"{study_name}_best.json")
    with open(best_path, "w") as f:
        json.dump(best_summary, f, indent=2)

    logger.info(f"Best params: {study.best_params}")
    logger.info(f"Best value: {study.best_value}")
    logger.info(f"Saved to: {best_path}")

    return best_summary