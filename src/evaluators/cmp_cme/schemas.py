from dataclasses import dataclass
from src.schemas import PromptLogprob


# Define experiment result structure
@dataclass(frozen=True)
class ExperimentResult:
    eval_id: int
    passage_id: str
    label: int
    question: str
    answer: str
    prompt_logprob: list[dict[str, PromptLogprob] | None]
    prefix_length: int
    ok: bool
    error: str | None = None
