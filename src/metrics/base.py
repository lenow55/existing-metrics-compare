from typing import Callable, Protocol, runtime_checkable

from openai.types.chat.chat_completion_token_logprob import TopLogprob

from src.schemas import MetricOutput, PromptLogprob, TextUnitMetric

# Тип одного шага: список top-k токенов на конкретной позиции.
# Первый элемент — фактически выбранный токен.
LogprobStep = list[PromptLogprob | TopLogprob]

# Редьюсер: по текущему и предыдущему шагу возвращает значение метрики либо None,
# если шаг нужно пропустить. Предыдущий шаг = None для первой позиции в logprobs.
# Предыдущий шаг сквозной по logprobs, в т.ч. на границе префикс→ответ.
StepValueFn = Callable[[LogprobStep, LogprobStep | None], float | None]


@runtime_checkable
class MetricSignature(Protocol):
    def __call__(
        self,
        *,
        logprobs: list[LogprobStep],
        context: str,
        question: str,
        prefix_length: int,
    ) -> MetricOutput: ...


METRICS_HUB: dict[str, MetricSignature] = {}


def register(id: str, f_metric: MetricSignature):
    if id in METRICS_HUB:
        raise RuntimeError(f"METRICS_HUB overrided by {id}")
    METRICS_HUB[id] = f_metric


def _head_token(step: LogprobStep | None) -> str | None:
    """Возвращает строковое представление фактически выбранного (первого) токена шага."""
    if not step:
        return None
    head = step[0]
    if isinstance(head, dict):
        return str(head["decoded_token"])
    return str(head.token)


def map_token_metric(
    *,
    logprobs: list[LogprobStep],
    context: str,
    question: str,
    prefix_length: int,
    value_fn: StepValueFn,
) -> MetricOutput:
    """
    Универсальный маппер логпробов промпта/ответа в MetricOutput.

    Промпт построен по схеме (см. src/evaluators/cmp-cme/logprob.py):
        <chat-template prefix> context: {context}\\nquestion: {question} <chat-template suffix>
    После prefix_length идут токены ответа ассистента.

    Алгоритм:
      1. Реконструируем текст префикса конкатенацией decoded_token фактически
         выбранных токенов и параллельно строим массив char_to_token такой,
         что char_to_token[i] = индекс токена, которому принадлежит символ i.
         Пример для токенов ["aa", "bbb", "cc"]:
            reconstructed = "aabbbcc"
            char_to_token = [0,0,1,1,1,2,2]
      2. По маркерам "context: " и "\\nquestion: " находим символьные срезы
         context и question внутри реконструированной строки.
      3. Срезы char_to_token по этим интервалам → множества индексов токенов
         для context / question. Всё, что не попало ни туда, ни туда — instruct.
      4. Дополнительно для каждого токена в контексте/вопросе считаем position
         как смещения первого и последнего символа токена, попавших в срез,
         относительно начала сегмента.
      5. Токены после prefix_length уходят в answer без position.
      6. Значение метрики на каждом шаге считается через value_fn(step, prev).
    """
    output = MetricOutput(instruct=[], context=[], question=[], answer=[])

    # 1. Реконструкция текста префикса + карта символ→токен.
    prefix_steps = logprobs[:prefix_length]
    token_text: list[str | None] = []
    decoded_buf: list[str] = []
    char_to_token: list[int] = []
    for idx, step in enumerate(prefix_steps):
        tok = _head_token(step)
        token_text.append(tok)
        if tok is None:
            continue
        decoded_buf.append(tok)
        char_to_token.extend([idx] * len(tok))
    reconstructed = "".join(decoded_buf)

    # 2. Положения context и question внутри реконструированной строки.
    ctx_marker = "context: "
    q_marker = "\nquestion: "
    ctx_pos = reconstructed.find(ctx_marker)
    if ctx_pos >= 0:
        ctx_start = ctx_pos + len(ctx_marker)
        ctx_end = ctx_start + len(context)
    else:
        ctx_start = ctx_end = -1

    q_pos = reconstructed.find(q_marker, ctx_end if ctx_end >= 0 else 0)
    if q_pos >= 0:
        q_start = q_pos + len(q_marker)
        q_end = q_start + len(question)
    else:
        q_start = q_end = -1

    # 3. Слайсы карты символ→токен и множества индексов для быстрых проверок.
    ctx_slice = char_to_token[ctx_start:ctx_end] if ctx_start >= 0 else []
    q_slice = char_to_token[q_start:q_end] if q_start >= 0 else []
    ctx_token_ids = set(ctx_slice)
    q_token_ids = set(q_slice)

    # Для пограничных токенов считаем position как (first_char, last_char + 1)
    # относительно начала соответствующего сегмента.
    def _positions_from_slice(
        char_slice: list[int],
    ) -> dict[int, tuple[int, int]]:
        positions: dict[int, tuple[int, int]] = {}
        for offset, tok_idx in enumerate(char_slice):
            if tok_idx not in positions:
                positions[tok_idx] = (offset, offset + 1)
            else:
                lo, _ = positions[tok_idx]
                positions[tok_idx] = (lo, offset + 1)
        return positions

    ctx_positions = _positions_from_slice(ctx_slice)
    q_positions = _positions_from_slice(q_slice)

    # 4. Распределяем токены префикса по сегментам.
    for idx, tok in enumerate(token_text):
        if tok is None:
            continue
        prev_step = logprobs[idx - 1] if idx > 0 else None
        value = value_fn(prefix_steps[idx], prev_step)
        if value is None:
            continue

        if idx in ctx_token_ids:
            unit: TextUnitMetric = {
                "value": value,
                "index": idx,
                "text_unit": tok,
                "position": ctx_positions[idx],
            }
            output.context.append(unit)
        elif idx in q_token_ids:
            unit = {
                "value": value,
                "index": idx,
                "text_unit": tok,
                "position": q_positions[idx],
            }
            output.question.append(unit)
        else:
            unit = {
                "value": value,
                "index": idx,
                "text_unit": tok,
            }
            output.instruct.append(unit)

    # 5. Ответ ассистента.
    for offset, step in enumerate(logprobs[prefix_length:]):
        tok = _head_token(step)
        if tok is None:
            continue
        abs_idx = prefix_length + offset
        prev_step = logprobs[abs_idx - 1] if abs_idx > 0 else None
        value = value_fn(step, prev_step)
        if value is None:
            continue
        output.answer.append(
            {
                "value": value,
                "index": abs_idx,
                "text_unit": tok,
            }
        )

    return output
