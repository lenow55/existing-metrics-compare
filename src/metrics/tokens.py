import numpy as np
from openai.types.chat.chat_completion_token_logprob import TopLogprob

from src.metrics.base import LogprobStep, map_token_metric, register
from src.schemas import MetricOutput, PromptLogprob


def _logprob(item: PromptLogprob | TopLogprob) -> float:
    """Достаёт logprob единообразно из TypedDict и pydantic-модели."""
    if isinstance(item, dict):
        return float(item["logprob"])
    return float(item.logprob)


def step_token_ll(step: LogprobStep) -> float | None:
    """
    Log Likelihood — logprob фактически выбранного (первого) токена шага.
    """
    if not step:
        return None
    return _logprob(step[0])


def step_token_entropy(step: LogprobStep) -> float | None:
    """
    Рассчитывает энтропию Шеннона (в битах) на основе Top-K logprobs шага.
    Формула: H = - sum(p * log2(p)).
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


def calculate_token_ll(
    *,
    logprobs: list[LogprobStep],
    context: str,
    question: str,
    prefix_length: int,
) -> MetricOutput:
    """Log Likelihood по токенам — обёртка над универсальным маппером."""
    return map_token_metric(
        logprobs=logprobs,
        context=context,
        question=question,
        prefix_length=prefix_length,
        value_fn=step_token_ll,
    )


def calculate_token_entropy(
    *,
    logprobs: list[LogprobStep],
    context: str,
    question: str,
    prefix_length: int,
) -> MetricOutput:
    """Энтропия Шеннона по токенам — обёртка над универсальным маппером."""
    return map_token_metric(
        logprobs=logprobs,
        context=context,
        question=question,
        prefix_length=prefix_length,
        value_fn=step_token_entropy,
    )


register(id="token_ll", f_metric=calculate_token_ll)
register(id="token_entropy", f_metric=calculate_token_entropy)
