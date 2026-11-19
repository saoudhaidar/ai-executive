from pathlib import Path

from eval.run_eval import main


ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    raise SystemExit(main(default_cases=ROOT / "tests" / "eval_set.json"))
