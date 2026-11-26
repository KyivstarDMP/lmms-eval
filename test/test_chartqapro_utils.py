import pytest

pytest.importorskip("anls")

from lmms_eval.tasks.chartqapro import utils  # noqa: E402


def _doc(question_type, questions, answers, paragraph="", year=None):
    return {
        "Question Type": question_type,
        "Question": questions,
        "Answer": answers,
        "Paragraph": paragraph,
        "Year": year or ["NO"] * len(questions),
    }


def test_each_question_type_gets_its_own_answer_format():
    multi_choice = utils.chartqapro_doc_to_text(_doc("Multi Choice", ["Which? a) x b) y"], ["b"]))
    assert "one of the options letters only: a, b, c or d" in multi_choice
    assert "Do not generate units" not in multi_choice
    assert multi_choice.endswith("Question: Which? a) x b) y")

    fact = utils.chartqapro_doc_to_text(_doc("Fact Checking", ["Sales grew."], ["True"]))
    assert "either true or false" in fact

    factoid = utils.chartqapro_doc_to_text(_doc("Factoid", ["How many?"], ["3"]))
    assert "a single word, number, or phrase" in factoid and "Do not generate units" in factoid
    assert "your answer should be unanswerable" in factoid


def test_conversation_history_and_paragraph_reach_the_prompt():
    doc = _doc("Conversational", ["First?", "Second?"], ["1", "2"], paragraph="Chart notes.")
    prompt = utils.chartqapro_doc_to_text(doc)
    assert prompt.startswith("Chart notes.\nYou are given a multi-turn conversation")
    assert prompt.endswith("Conversation: ['First?', '1'] Question: Second?")


def test_cot_prompt_asks_for_a_final_sentence():
    prompt = utils.chartqapro_doc_to_text(_doc("Factoid", ["How many?"], ["3"]), {"prompt_style": "cot"})
    assert "think step-by-step" in prompt and 'like "The answer is X"' in prompt
    with pytest.raises(ValueError, match="prompt_style"):
        utils.chartqapro_doc_to_text(_doc("Factoid", ["How many?"], ["3"]), {"prompt_style": "pot"})


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Bars are 2 and 1.\nThe answer is 3.", "3"),
        ("The answer is 2... no.\nThe answer is **b**.", "b"),
        ("So the final answer is: ['Asia', 'Europe']", "['Asia', 'Europe']"),
        ("Step 1.\nunanswerable", "unanswerable"),
        ("", ""),
    ],
)
def test_cot_scores_the_final_answer_only(response, expected):
    assert utils._final_answer(response) == expected


def test_cot_and_direct_share_the_scorer():
    doc = _doc("Multi Choice", ["Which?"], ["b"])
    assert utils.chartqapro_process_results(doc, ["B"]) == {"relaxed_overall": 1.0, "relaxed_multi_choice": 1.0, "relaxed_answerable": 1.0, "false_unanswerable": 0.0}
    assert utils.chartqapro_cot_process_results(doc, ["x is larger.\nThe answer is b."])["relaxed_overall"] == 1.0
    assert utils.chartqapro_process_results(doc, ["x is larger.\nThe answer is b."])["relaxed_overall"] == 0.0


def test_unanswerable_questions_are_scored_apart():
    unanswerable = _doc("Factoid", ["How many in 2031?"], ["Unanswerable"])
    assert utils.chartqapro_process_results(unanswerable, ["unanswerable"]) == {"relaxed_overall": 1.0, "relaxed_factoid": 1.0, "relaxed_unanswerable": 1.0}
    assert utils.chartqapro_process_results(unanswerable, ["42"])["relaxed_unanswerable"] == 0.0

    answerable = _doc("Factoid", ["How many?"], ["3"])
    refused = utils.chartqapro_process_results(answerable, ["['Unanswerable']"])
    assert refused["relaxed_answerable"] == 0.0 and refused["false_unanswerable"] == 1.0
    assert "relaxed_unanswerable" not in refused
