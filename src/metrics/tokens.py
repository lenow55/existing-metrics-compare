import numpy as np
from openai.types.chat.chat_completion_token_logprob import TopLogprob

from src.metrics.base import LogprobStep, register
from src.schemas import PromptLogprob, TextUnitMetric


def _logprob(item: PromptLogprob | TopLogprob) -> float:
    """Достаёт logprob единообразно из TypedDict и pydantic-модели."""
    if isinstance(item, dict):
        return float(item["logprob"])
    return float(item.logprob)


def _token(item: PromptLogprob | TopLogprob) -> str:
    """Достаёт token-str единообразно из TypedDict и pydantic-модели."""
    if isinstance(item, dict):
        return item["decoded_token"]
    return item.token


def step_token_ll(logprobs: list[LogprobStep]) -> list[TextUnitMetric]:
    """
    Log Likelihood — logprob фактически выбранного (первого) токена шага.
    Предыдущий шаг не используется.
    """
    result: list[TextUnitMetric] = []
    for idx, logprob in enumerate(logprobs):
        value = _logprob(logprob[0])
        result.append(
            TextUnitMetric(
                value=value,
                index=idx,
                text_unit=_token(logprob[0]),
            )
        )
    return result


def step_token_nll(logprobs: list[LogprobStep]) -> list[TextUnitMetric]:
    """
    Negative Log Likelihood — logprob фактически выбранного (первого) токена шага.
    Умноженный на -1.
    Предыдущий шаг не используется.
    """
    result: list[TextUnitMetric] = []
    for idx, logprob in enumerate(logprobs):
        value = _logprob(logprob[0])
        result.append(
            TextUnitMetric(
                value=value * (-1),
                index=idx,
                text_unit=_token(logprob[0]),
            )
        )
    return result


def step_token_inflection(step: LogprobStep, prev: LogprobStep | None) -> float | None:
    """
    Inflection point — разница LL текущего и предыдущего токенов: LL_t - LL_{t-1}.
    Для самой первой позиции возвращает None (нет предыдущего токена).
    """
    if not step or not prev:
        return None
    return _logprob(step[0]) - _logprob(prev[0])


def step_token_entropy(
    step: LogprobStep, _prev: LogprobStep | None = None
) -> float | None:
    """
    Рассчитывает энтропию Шеннона (в битах) на основе Top-K logprobs шага.
    Формула: H = - sum(p * log2(p)). Предыдущий шаг не используется.
    """
    if not step:
        return None

    # OpenAI возвращает logprob (натуральный логарифм), конвертируем в вероятность
    probs_arr = np.array([np.exp(_logprob(item)) for item in step], dtype=float)
    total = float(np.sum(probs_arr))
    if total <= 0:
        return None

    # Нормализуем вероятности, так как у нас только Top-K, а не полный словарь.
    # Это даёт аппроксимацию энтропии.
    probs_norm = probs_arr / total

    entropy = -float(np.sum(probs_norm * np.log2(probs_norm + 1e-9)))
    return entropy


register(id="token_ll", f_metric=step_token_ll)
register(id="token_nll", f_metric=step_token_nll)
# register(id="token_entropy", f_metric=calculate_token_entropy)
# register(id="token_inflection", f_metric=calculate_token_inflection)
