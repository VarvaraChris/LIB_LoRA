import argparse
import json

import optuna


def build_default_study_name(model, dataset, ft_strategy):
    model_name = model.replace("/", "_")
    return f"{model_name}__{dataset}__{ft_strategy}"


def main():
    parser = argparse.ArgumentParser(description="Show current best Optuna trial from a study DB.")
    parser.add_argument("--study_name", type=str, default=None, help="Exact Optuna study name")
    parser.add_argument("--storage", type=str, required=True, help="Optuna storage URL, e.g. sqlite:///path/to.db")
    parser.add_argument("--model", type=str, default=None, help="Model name to build study_name automatically")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name to build study_name automatically")
    parser.add_argument("--ft_strategy", type=str, default=None, help="FT strategy to build study_name automatically")
    parser.add_argument("--show_top", type=int, default=5, help="How many completed trials to print")
    args = parser.parse_args()

    study_name = args.study_name
    if study_name is None:
        if not all([args.model, args.dataset, args.ft_strategy]):
            raise ValueError(
                "Pass either --study_name or all of --model, --dataset, --ft_strategy"
            )
        study_name = build_default_study_name(args.model, args.dataset, args.ft_strategy)

    study = optuna.load_study(
        study_name=study_name,
        storage=args.storage,
    )

    completed_trials = [trial for trial in study.trials if trial.value is not None]

    print(f"study_name: {study.study_name}")
    print(f"direction: {study.direction.name}")
    print(f"total_trials: {len(study.trials)}")
    print(f"completed_trials: {len(completed_trials)}")
    print()

    best_trial = study.best_trial
    print("current_best:")
    print(f"  trial_number: {best_trial.number}")
    print(f"  value: {best_trial.value}")
    print(f"  params: {json.dumps(best_trial.params, ensure_ascii=False)}")
    if best_trial.user_attrs:
        print(f"  user_attrs: {json.dumps(best_trial.user_attrs, ensure_ascii=False)}")
    print()

    reverse = study.direction.name == "MAXIMIZE"
    top_trials = sorted(completed_trials, key=lambda t: t.value, reverse=reverse)[: args.show_top]

    print(f"top_{len(top_trials)}_completed_trials:")
    for trial in top_trials:
        print(
            f"  trial={trial.number} value={trial.value} params={json.dumps(trial.params, ensure_ascii=False)}"
        )


if __name__ == "__main__":
    main()