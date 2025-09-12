from doc_explainer.type import ExplainableAnswer


def test_explainable_answer():
    # Given
    answer = "answer"
    page = 0
    bbox = [1.0, 2.0, 3.0, 4.0]

    # When
    exp_answer = ExplainableAnswer(answer=answer, page=page, bbox=bbox)

    # Then
    assert exp_answer.answer == answer
    assert exp_answer.page == page
    assert exp_answer.bbox == bbox
    assert str(exp_answer) == f"Answer: {answer} | Page: {page} | BBox: {bbox}"
