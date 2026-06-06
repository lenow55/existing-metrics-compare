import numpy as np
from openai.types.chat.chat_completion_token_logprob import TopLogprob

from src.metrics.base import register
from src.schemas import LogprobStep, PromptLogprob, TextUnitMetric


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


def step_token_entropy(logprobs: list[LogprobStep]) -> list[TextUnitMetric]:
    """
    Рассчитывает энтропию Шеннона
    """
    result: list[TextUnitMetric] = []
    for idx, step in enumerate(logprobs):
        actual_token = _token(step[0])
        step_logprobs = [_logprob(i) for i in step]
        step_logprobs_a = np.array(step_logprobs, dtype=np.float64)

        K = len(step_logprobs)
        if K <= 1:
            # Нет неопределенности, если токен всего один
            result.append(
                TextUnitMetric(
                    value=0.0,
                    index=idx,
                    text_unit=actual_token,
                )
            )

        # 1. Нахождение максимального логпроба (M)
        max_logprob = np.max(step_logprobs_a)

        # 2. Вычисление знаменателя сдвинутых экспонент (Log-Sum-Exp)
        # Вычитание max_logprob защищает от underflow/overflow
        lse = max_logprob + np.log(np.sum(np.exp(step_logprobs_a - max_logprob)))

        # 3. Получение нормализованных логпробов
        normalized_logprobs = step_logprobs_a - lse

        # 4. Перевод в вероятности (безопасно, так как максимум равен 0)
        normalized_probs = np.exp(normalized_logprobs)

        # 5. Расчет энтропии Шеннона
        entropy = -np.sum(normalized_probs * normalized_logprobs)

        # 6. Нормализация до [0, 1]
        entropy = entropy / np.log(K)

        result.append(
            TextUnitMetric(
                value=float(entropy),
                index=idx,
                text_unit=actual_token,
            )
        )

    return result


register(id="token_ll", f_metric=step_token_ll)
register(id="token_nll", f_metric=step_token_nll)
register(id="token_entropy", f_metric=step_token_entropy)
# register(id="token_inflection", f_metric=calculate_token_inflection)
