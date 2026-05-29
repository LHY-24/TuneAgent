from tuneagent.src.reward_score.tuneagent_score import (
    compute_score_answer,
    compute_score_format,
)


def test_bool_reward_accepts_camera_ready_answer_format():
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

    assert compute_score_answer(solution, ground_truth) == 1.0


def test_format_reward_requires_structured_answer_block():
    solution = (
        "<|im_start|>assistant\n"
        "<think>Reasoning.</think>\n"
        "<answer>[SMP - increase]</answer>"
        "<|im_end|>"
    )

    assert compute_score_format(solution) == 0.55
