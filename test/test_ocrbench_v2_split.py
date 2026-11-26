import pytest

datasets = pytest.importorskip("datasets")
utils = pytest.importorskip("lmms_eval.tasks.ocrbench_v2.utils")


def test_each_language_task_keeps_only_its_questions():
    data = datasets.Dataset.from_dict({"type": ["text spotting en", "full-page OCR cn", "math QA en"], "id": [0, 1, 2]})
    assert utils.ocrbench_v2_process_docs_en(data)["id"] == [0, 2]
    assert utils.ocrbench_v2_process_docs_cn(data)["id"] == [1]


def test_language_tasks_score_like_the_combined_one():
    results = [
        {"question_type": "text spotting en", "score": 1.0},
        {"question_type": "math QA en", "score": 0.0},
        {"question_type": "full-page OCR cn", "score": 0.5},
    ]
    english = [row for row in results if row["question_type"].endswith(" en")]
    chinese = [row for row in results if row["question_type"].endswith(" cn")]
    assert utils.ocrbench_v2_aggregate_accuracy_en(english, None) == utils.ocrbench_v2_aggregate_accuracy_en(results, None) == 0.5
    assert utils.ocrbench_v2_aggregate_accuracy_cn(chinese, None) == utils.ocrbench_v2_aggregate_accuracy_cn(results, None) == 0.5
