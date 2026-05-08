import json
import logging
from logging import config as log_config_m

import torch
from httpx import AsyncClient, Timeout
from openai import AsyncOpenAI

from src.config import EmbedLLMConfig, LLMConfig

logger = logging.getLogger(__name__)


def configure_logging(logging_conf_file: str):
    with open(logging_conf_file) as l_f:
        logging_config_dict = json.loads(l_f.read())
        log_config_m.dictConfig(logging_config_dict)


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


async def calculate_similarity(
    idx: str,
    reference: str,
    answer: str,
    client: AsyncOpenAI,
    config: EmbedLLMConfig,
) -> tuple[str, float]:
    """
    Подсчитывает близость ответа ллм к эталону
    """
    logger.debug(f"Start Embed request id {idx}")

    reference_i = get_detailed_instruct(
        task_description="Это эталонный ответ. С ним сравнивают предполагаемый ответ для получения оценки правильности.",
        query=reference,
    )
    answer_i = get_detailed_instruct(
        task_description="Это предполагаемый ответ. Его сравнивают с эталонным ответом для получения оценки правильности.",
        query=answer,
    )

    response = await client.embeddings.create(
        input=[answer_i, reference_i],
        model=config.model,
        extra_body=config.extra_body,
        **config.params_extra,
    )
    embeddings = torch.tensor([o.embedding for o in response.data])
    scores = embeddings[:1] @ embeddings[1:].T
    score: float = scores.tolist()[0][0]

    return idx, score


def create_openai_client(config: LLMConfig) -> AsyncOpenAI:
    if config.proxy_url:
        http_client = AsyncClient(proxy=config.proxy_url)
    else:
        http_client = AsyncClient()

    client = AsyncOpenAI(
        api_key=config.api_key.get_secret_value(),
        base_url=config.base_url,
        timeout=Timeout(config.timeout),
        http_client=http_client,
    )
    return client
