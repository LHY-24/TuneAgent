import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tuneagent.src.reward_score.tuneagent_score import (
    compute_score_answer,
    compute_score_format,
)


def main() -> None:
    solution = (
        "<|im_start|>assistant\n"
        "<think>Inspect the target and candidate configs.</think>\n"
        "<answer>[SMP - increase]\n"
        "[DEBUG_FS - cannot determine impact without specific context]</answer>"
        "<|im_end|>"
    )
    ground_truth = (
        'Bool\t[{"config": "smp", "value": 2}, '
        '{"config": "debug_fs", "value": -1}]'
    )

    answer_score = compute_score_answer(solution, ground_truth)
    format_score = compute_score_format(solution)

    if answer_score != 1.0:
        raise SystemExit(f"Unexpected answer score: {answer_score}")
    if format_score != 0.55:
        raise SystemExit(f"Unexpected format score: {format_score}")

    print("TuneAgent sanity check passed.")
    print(f"answer_score={answer_score}")
    print(f"format_score={format_score}")


if __name__ == "__main__":
    main()
