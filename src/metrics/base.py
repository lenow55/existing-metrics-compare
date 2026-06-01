from typing import Callable, Protocol, runtime_checkable

from openai.types.chat import ChatCompletionTokenLogprob

from src.schemas import (
    LogprobParts,
    LogprobStep,
    MetricOutput,
    PromptLogprob,
    TextUnitMetric,
)

# Тип одного шага: список top-k токенов на конкретной позиции.
# Первый элемент — фактически выбранный токен.

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


def map_logprobs2parts(
    *,
    prompt_logprob: list[None | dict[str, PromptLogprob]],
    top_logprob: list[ChatCompletionTokenLogprob],
    context: str,
    question: str,
    prefix_length: int,
):
    # 1. Реконструкция текста промпта + карта символ→токен.
    prefix_steps = prompt_logprob[:prefix_length]
    token_text: list[str] = []
    decoded_buf: list[str] = []
    char_to_token: list[int] = []

    for idx, step in enumerate(prefix_steps):
        if step is None:
            continue
        logprob = next(iter(step.values()))
        tok = logprob["decoded_token"]
        token_text.append(tok)
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
    #
    # 3. Слайсы карты символ→токен и множества индексов для быстрых проверок.
    ctx_slice = char_to_token[ctx_start:ctx_end] if ctx_start >= 0 else []
    q_slice = char_to_token[q_start:q_end] if q_start >= 0 else []
    ctx_token_ids = set(ctx_slice)
    q_token_ids = set(q_slice)

    output = LogprobParts(instruct=[], context=[], question=[], answer=[])
    # 4. Распределяем токены префикса по сегментам.
    first_not_instruct = min(min(ctx_token_ids), min(q_token_ids))
    for idx, tok in enumerate(token_text):
        if idx < first_not_instruct:
            probs = prefix_steps[idx]
            if probs is None:
                continue
            output.instruct.append(list(probs.values()))
        elif idx in ctx_token_ids:
            probs = prefix_steps[idx]
            if probs is None:
                continue
            output.context.append(list(probs.values()))
        elif idx in q_token_ids:
            probs = prefix_steps[idx]
            if probs is None:
                continue
            output.question.append(list(probs.values()))

    # 5. Ответ ассистента из промптлогпроб
    for step in prompt_logprob[prefix_length:]:
        if step is None:
            continue
        probs = list(step.values())
        output.answer.append(probs)

    # 6. Ответ ассистента из топ пробов
    for step in top_logprob:
        probs = step.top_logprobs
        output.answer.append(probs)
