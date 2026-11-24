import pytest

nltk = pytest.importorskip("nltk")

from lmms_eval.tasks._task_utils.nltk_distance import allow_long_inputs  # noqa: E402


def test_a_page_of_text_can_be_compared():
    allow_long_inputs()

    assert nltk.edit_distance("a" * 2500, "b") == 2500
