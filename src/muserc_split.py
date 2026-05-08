"""
Разбор MuSeRC-элементов на эталон и «генерации» по правилам:

1. Берём только вопросы, где есть хотя бы один ответ с label == 1.
2. Один случайный правильный ответ → эталон (reference).
3. Оставшиеся правильные ответы → успешные генерации.
4. Берём столько же неправильных ответов (label == 0), сколько было «лишних»
   правильных (без замены через random.sample); если таких меньше, кладём все
   доступные неверные.

Плоский вывод: flat_generation_records / iter_flat_generation_rows — по одному
словарю на каждую генерацию с полем label (1 корректная, 0 провальная).
"""

import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, TypedDict

from src.schemas import Answer, Question, ReadingComprehensionItem


class FlatGenerationRow(TypedDict):
    """Одна строка: модель считает `answer` корректной (1) или нет (0) относительно эталона."""

    passage_id: str
    question_idx: int
    question: str
    reference: str
    reference_answer_idx: int
    answer: str
    answer_idx: int
    label: int


@dataclass(frozen=True, slots=True)
class SplitAnswers:
    """Результат разбиения одного вопроса."""

    reference: Answer
    generations_correct: list[Answer]
    generations_failed: list[Answer]


@dataclass(frozen=True, slots=True)
class ProcessedQuestion:
    passage_item_idx: int
    passage_text: str
    question_idx: int
    question_text: str
    split: SplitAnswers


def _answers_by_label(question: Question) -> tuple[list[Answer], list[Answer]]:
    correct = [a for a in question.answers if a.label == 1]
    incorrect = [a for a in question.answers if a.label != 1]
    return correct, incorrect


def split_question_answers(
    question: Question, rng: random.Random
) -> SplitAnswers | None:
    """
    Возвращает разбиение для одного вопроса или None, если нет ни одного
    верного ответа.
    """
    correct, incorrect = _answers_by_label(question)
    if not correct:
        return None

    correct_perm = correct.copy()
    rng.shuffle(correct_perm)
    reference = correct_perm[0]
    gens_ok = correct_perm[1:]
    need_failed = len(gens_ok)

    if need_failed <= 0:
        return SplitAnswers(
            reference=reference,
            generations_correct=[],
            generations_failed=[],
        )

    if len(incorrect) >= need_failed:
        failed = rng.sample(incorrect, need_failed)
    else:
        failed = incorrect.copy()
        rng.shuffle(failed)

    return SplitAnswers(
        reference=reference,
        generations_correct=gens_ok,
        generations_failed=failed,
    )


def process_reading_item(
    item: ReadingComprehensionItem,
    rng: random.Random | None = None,
) -> list[ProcessedQuestion]:
    """
    Обрабатывает один элемент датасета (passage + вопросы).
    """
    rng = rng or random.Random()
    passage = item.passage
    out: list[ProcessedQuestion] = []

    for q in passage.questions:
        split = split_question_answers(q, rng)
        if split is None:
            continue
        out.append(
            ProcessedQuestion(
                passage_item_idx=item.idx,
                passage_text=passage.text,
                question_idx=q.idx,
                question_text=q.question,
                split=split,
            )
        )

    return out


def iter_processed_batches(
    items: Iterator[ReadingComprehensionItem],
    rng: random.Random | None = None,
) -> Iterator[dict[str, Any]]:
    """
    Итерация по разобранным вопросам: для каждой строки один эталон, списки
    дополнительных правильных ответов и провальных генераций.
    """
    rng = rng or random.Random()
    for item in items:
        for pq in process_reading_item(item, rng=rng):
            sp = pq.split
            yield {
                "passage_id": str(pq.passage_item_idx),
                "passage": pq.passage_text,
                "question_idx": pq.question_idx,
                "question": pq.question_text,
                "reference_answer_idx": sp.reference.idx,
                "reference": sp.reference.text,
                "generation_correct": [
                    {"idx": a.idx, "text": a.text} for a in sp.generations_correct
                ],
                "generation_failed": [
                    {"idx": a.idx, "text": a.text} for a in sp.generations_failed
                ],
            }


def flat_rows_for_processed_question(pq: ProcessedQuestion) -> list[FlatGenerationRow]:
    """Плоские строки по одному варианту-«генерации» (эталон в каждой строке продублирован)."""
    sp = pq.split
    passage_id = str(pq.passage_item_idx)
    ref_text = sp.reference.text
    ref_idx = sp.reference.idx
    q_idx = pq.question_idx
    q_text = pq.question_text

    rows: list[FlatGenerationRow] = []

    for a in sp.generations_correct:
        rows.append(
            {
                "passage_id": passage_id,
                "question_idx": q_idx,
                "question": q_text,
                "reference": ref_text,
                "reference_answer_idx": ref_idx,
                "answer": a.text,
                "answer_idx": a.idx,
                "label": 1,
            }
        )

    for a in sp.generations_failed:
        rows.append(
            {
                "passage_id": passage_id,
                "question_idx": q_idx,
                "question": q_text,
                "reference": ref_text,
                "reference_answer_idx": ref_idx,
                "answer": a.text,
                "answer_idx": a.idx,
                "label": 0,
            }
        )

    return rows


def flat_generation_records(
    items: list[ReadingComprehensionItem],
    rng: random.Random | None = None,
) -> list[FlatGenerationRow]:
    """
    Сводный список плоских словарей по всем элементам датасета.
    Пустых строк нет там, где у вопроса только один правильный ответ (нет «генераций»).
    """
    rng = rng or random.Random()
    flat: list[FlatGenerationRow] = []
    for item in items:
        for pq in process_reading_item(item, rng=rng):
            flat.extend(flat_rows_for_processed_question(pq))
    return flat


def iter_flat_generation_rows(
    items: Iterator[ReadingComprehensionItem],
    rng: random.Random | None = None,
) -> Iterator[FlatGenerationRow]:
    """Ленивая версия flat_generation_records."""
    rng = rng or random.Random()
    for item in items:
        for pq in process_reading_item(item, rng=rng):
            yield from flat_rows_for_processed_question(pq)


if __name__ == "__main__":
    import json
    import sys

    sample = json.load(sys.stdin) if not sys.stdin.isatty() else None
    if sample is None:
        raise SystemExit("Передай JSON в stdin.")

    rng = random.Random(42)
    item = ReadingComprehensionItem.model_validate(sample)
    rows = flat_generation_records([item], rng=rng)
    json.dump(rows, sys.stdout, ensure_ascii=False, indent=2)
