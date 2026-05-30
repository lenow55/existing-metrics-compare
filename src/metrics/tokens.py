import numpy as np
from openai.types.chat.chat_completion_token_logprob import TopLogprob

from src.metrics.base import register
from src.schemas import MetricOutput, PromptLogprob, TextUnitMetric


def calculate_token_entropy(top_logprobs: list[PromptLogprob]) -> float:
    """
    Рассчитывает энтропию Шеннона (в битах) на основе Top-K logprobs.
    Формула: H = - sum(p * log2(p))
    """
    probs: list[float] = []
    for item in top_logprobs:
        # OpenAI возвращает logprob (натуральный логарифм), конвертируем в вероятность
        p = np.exp(item["logprob"])
        probs.append(p)

    probs_arr = np.array(probs, dtype=float)

    # Нормализуем вероятности, так как у нас только Top-K, а не полный словарь
    # Это дает аппроксимацию энтропии
    probs_norm = probs_arr / np.sum(probs_arr)

    # Считаем энтропию
    entropy = -np.sum(
        probs_norm * np.log2(probs_norm + 1e-9)
    )  # +1e-9 для защиты от log(0)
    if not isinstance(entropy, float):
        raise RuntimeError(f"Bad result type {type(entropy)}")
    return entropy


def _first_logprob(
    step: list[PromptLogprob | TopLogprob] | None,
) -> tuple[float, str] | None:
    """
    Возвращает (logprob, decoded_token) для реально выбранного токена на шаге.
    Это первый элемент списка top-k. Если шаг пустой/None — None.
    """
    if not step:
        return None
    head = step[0]
    # PromptLogprob (TypedDict) использует ключ 'decoded_token',
    # TopLogprob (pydantic) — атрибут 'token'.
    if isinstance(head, dict):
        return float(head["logprob"]), str(head["decoded_token"])
    return float(head.logprob), str(head.token)


def calculate_token_ll(
    logprobs: list[list[PromptLogprob | TopLogprob]],
    context: str,
    question: str,
    prefix_length: int,
) -> MetricOutput:
    """
    Возвращает Log Likelihood — logprob фактически выбранного (первого) токена
    на каждой позиции.

    Промпт построен по схеме:
        <chat-template prefix> context: {context}\\nquestion: {question} <chat-template suffix>
    Затем идёт ответ ассистента (его токены лежат в logprobs[prefix_length:]).
    """

    output: MetricOutput = MetricOutput(instruct=[], context=[], question=[], answer=[])

    # 1. Собираем для каждой позиции префикса decoded_token и диапазон
    #    символов [char_start, char_end) в восстановленной из токенов строке.
    prefix_steps = logprobs[:prefix_length]
    decoded_buf: list[str] = []
    token_spans: list[tuple[int, int]] = []  # позиция токена в восстановленной строке
    token_payload: list[tuple[float, str] | None] = []
    cursor = 0
    for step in prefix_steps:
        info = _first_logprob(step)
        token_payload.append(info)
        if info is None:
            token_spans.append((cursor, cursor))
            continue
        _, tok = info
        start = cursor
        end = cursor + len(tok)
        token_spans.append((start, end))
        decoded_buf.append(tok)
        cursor = end
    reconstructed = "".join(decoded_buf)

    # 2. Находим положение context и question внутри восстановленной строки.
    ctx_marker = "context: "
    q_marker = "\nquestion: "
    ctx_marker_pos = reconstructed.find(ctx_marker)
    if ctx_marker_pos >= 0:
        ctx_start = ctx_marker_pos + len(ctx_marker)
        ctx_end = ctx_start + len(context)
    else:
        ctx_start = ctx_end = -1

    q_marker_pos = reconstructed.find(q_marker, ctx_end if ctx_end >= 0 else 0)
    if q_marker_pos >= 0:
        q_start = q_marker_pos + len(q_marker)
        q_end = q_start + len(question)
    else:
        q_start = q_end = -1

    def _overlap(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int] | None:
        lo = max(a[0], b[0])
        hi = min(a[1], b[1])
        if hi <= lo:
            return None
        return (lo, hi)

    # 3. Распределяем токены префикса по сегментам instruct / context / question.
    for idx, (info, span) in enumerate(zip(token_payload, token_spans)):
        if info is None:
            continue
        lp, tok = info

        ctx_ov = _overlap(span, (ctx_start, ctx_end)) if ctx_start >= 0 else None
        q_ov = _overlap(span, (q_start, q_end)) if q_start >= 0 else None

        if ctx_ov is not None:
            unit: TextUnitMetric = {
                "value": lp,
                "index": idx,
                "text_unit": tok,
                "position": (ctx_ov[0] - ctx_start, ctx_ov[1] - ctx_start),
            }
            output.context.append(unit)
        elif q_ov is not None:
            unit = {
                "value": lp,
                "index": idx,
                "text_unit": tok,
                "position": (q_ov[0] - q_start, q_ov[1] - q_start),
            }
            output.question.append(unit)
        else:
            unit = {
                "value": lp,
                "index": idx,
                "text_unit": tok,
            }
            output.instruct.append(unit)

    # 4. Ответ ассистента — всё, что идёт после префикса.
    for offset, step in enumerate(logprobs[prefix_length:]):
        info = _first_logprob(step)
        if info is None:
            continue
        lp, tok = info
        unit = {
            "value": lp,
            "index": prefix_length + offset,
            "text_unit": tok,
        }
        output.answer.append(unit)

    return output


register(id="token_ll", f_metric=calculate_token_ll)
