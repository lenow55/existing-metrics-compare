from typing import TypedDict


class EvalIn(TypedDict):
    eval_id: int
    passage_id: str
    question: str
    reference: str
    answer: str
    label: int


class EvalOut(EvalIn):
    eval_score: float
