import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CSE382 Hotel Booking ML")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("analyze", help="data inspection tables and EDA figures")
    commands.add_parser("baselines", help="Milestone 1 baseline CV experiments")
    commands.add_parser("ablation", help="leakage ablation (duplicates, DepositType)")
    commands.add_parser("cluster", help="clustering experiments")

    train = commands.add_parser("train", help="baseline + tuning, saves artifacts")
    train.add_argument("--task", required=True, choices=["regression", "classification"])

    predict = commands.add_parser("predict", help="predict using the saved pipeline")
    predict.add_argument("--task", required=True, choices=["regression", "classification"])
    predict.add_argument("--input", help="JSON file with booking fields")
    predict.add_argument("--example", action="store_true", help="print an example JSON")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "analyze":
        from src.data.eda import main as run_eda
        from src.data.load_data import main as run_inspection

        run_inspection()
        run_eda()
    elif args.command == "baselines":
        from src.evaluation.baselines import main as run
        run()
    elif args.command == "ablation":
        from src.classification.ablation import main as run
        run()
    elif args.command == "cluster":
        from src.clustering.train import main as run
        run()
    elif args.command == "train":
        if args.task == "regression":
            from src.regression.train import run_experiments
        else:
            from src.classification.train import run_experiments
        run_experiments()
    elif args.command == "predict":
        from src.inference.predict import run_cli
        run_cli(args.task, args.input, args.example)


if __name__ == "__main__":
    main()
