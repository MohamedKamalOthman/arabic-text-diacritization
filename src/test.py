import argparse

from config import CONFIG
from tester import Tester

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        choices=CONFIG["models"],
        default=CONFIG["test_model_name"],
        help="Model to use",
    )
    parser.add_argument(
        "--model_path",
        default=CONFIG["test_model_path"],
        type=str,
        help="Path to model snapshot",
    )
    args = parser.parse_args()

    tester = Tester(model_name=args.model_name, model_path=args.model_path)
    tester.test_all(test_directory=CONFIG["test_data_directory"])
