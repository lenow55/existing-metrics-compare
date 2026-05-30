from typing import Protocol, runtime_checkable

from openai.types.chat.chat_completion_token_logprob import TopLogprob

from src.schemas import MetricOutput, PromptLogprob


@runtime_checkable
class MetricSignature(Protocol):
    def __call__(
        self,
        *,
        logprobs: list[list[PromptLogprob | TopLogprob]],
        context: str,
        question: str,
        prefix_length: int,
    ) -> MetricOutput: ...


METRICS_HUB: dict[str, MetricSignature] = {}


def register(id: str, f_metric: MetricSignature):
    if id in METRICS_HUB:
        raise RuntimeError(f"METRICS_HUB overrided by {id}")
    METRICS_HUB[id] = f_metric
