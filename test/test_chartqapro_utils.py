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
