from typing import Callable, Protocol, runtime_checkable

from openai.types.chat.chat_completion_token_logprob import TopLogprob

from src.schemas import MetricOutput, PromptLogprob, TextUnitMetric

# Тип одного шага: список top-k токенов на конкретной позиции.
# Первый элемент — фактически выбранный токен.
LogprobStep = list[PromptLogprob | TopLogprob]

# Редьюсер: по шагу возвращает значение метрики либо None, если шаг нужно пропустить.
StepValueFn = Callable[[LogprobStep], float | None]


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


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int] | None:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    if hi <= lo:
        return None
    return (lo, hi)


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

    Функция:
      1. Реконструирует текст префикса конкатенацией decoded_token фактически
         выбранных токенов и запоминает символьный интервал каждого токена.
      2. По маркерам "context: " и "\\nquestion: " находит расположение context
         и question в реконструированной строке.
      3. Каждому токену префикса по пересечению символьного интервала ставит
         в соответствие сегмент: context / question / иначе instruct.
      4. Все токены после prefix_length уходят в answer.
      5. Значение метрики на каждом шаге считается переданной value_fn.
         Если value_fn вернула None — шаг пропускается.
    """
    output = MetricOutput(instruct=[], context=[], question=[], answer=[])

    # 1. Префикс: восстановим текст и интервалы токенов.
    prefix_steps = logprobs[:prefix_length]
    token_text: list[str | None] = []
    token_spans: list[tuple[int, int]] = []
    cursor = 0
    for step in prefix_steps:
        tok = _head_token(step)
        token_text.append(tok)
        if tok is None:
            token_spans.append((cursor, cursor))
            continue
        token_spans.append((cursor, cursor + len(tok)))
        cursor += len(tok)
    reconstructed = "".join(t for t in token_text if t is not None)

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

    # 3. Распределяем токены префикса по сегментам.
    for idx, (tok, span) in enumerate(zip(token_text, token_spans)):
        if tok is None:
            continue
        value = value_fn(prefix_steps[idx])
        if value is None:
            continue

        ctx_ov = _overlap(span, (ctx_start, ctx_end)) if ctx_start >= 0 else None
        q_ov = _overlap(span, (q_start, q_end)) if q_start >= 0 else None

        if ctx_ov is not None:
            unit: TextUnitMetric = {
                "value": value,
                "index": idx,
                "text_unit": tok,
                "position": (ctx_ov[0] - ctx_start, ctx_ov[1] - ctx_start),
            }
            output.context.append(unit)
        elif q_ov is not None:
            unit = {
                "value": value,
                "index": idx,
                "text_unit": tok,
                "position": (q_ov[0] - q_start, q_ov[1] - q_start),
            }
            output.question.append(unit)
        else:
            unit = {
                "value": value,
                "index": idx,
                "text_unit": tok,
            }
            output.instruct.append(unit)

    # 4. Ответ ассистента.
    for offset, step in enumerate(logprobs[prefix_length:]):
        tok = _head_token(step)
        if tok is None:
            continue
        value = value_fn(step)
        if value is None:
            continue
        output.answer.append(
            {
                "value": value,
                "index": prefix_length + offset,
                "text_unit": tok,
            }
        )

    return output
