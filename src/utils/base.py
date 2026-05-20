import json
import logging
from logging import config as log_config_m

import torch
from httpx import AsyncClient, Timeout
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import TypeAdapter

from src.config import ChatLLMConfig, EmbedLLMConfig, LLMConfig
from src.schemas import PromptLogprob

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


async def calculate_prompt_logprobs(
    messages: list[ChatCompletionMessageParam],
    client: AsyncOpenAI,
    config: ChatLLMConfig,
) -> tuple[str, list[None | dict[str, PromptLogprob]]]:
    """
    Отправляет запрос в LLM и возвращает сгенерированный ответ вместе с логпробами промпта.

    Функция выполняет асинхронный вызов к API модели, запрашивая генерацию текста
    и информацию о вероятностях (logprobs) для токенов входного промпта (топ-5 вариантов).
    Для контроля конкурентности используется семафор.

    Args:
        query (str): Пользовательский запрос (промпт) для отправки в модель.
        client (AsyncOpenAI): Асинхронный клиент OpenAI.
        config (LLMConfig): Конфигурация модели, содержащая дополнительные параметры
            запроса (`extra_body` и `params_extra`).
        model (str): Название используемой LLM модели (например, 'gpt-4o').

    Returns:
        tuple[str, list[None | dict[str, PromptLogprob]]]: Кортеж, состоящий из двух элементов:
            - Строка со сгенерированным ответом модели.
            - Список вероятностей (логпробов) для каждого токена исходного промпта.

    Raises:
        RuntimeError: Если API модели не вернуло данные о логпробах промпта
            в поле `model_extra` или само поле `model_extra` отсутствует.
    """
    # INFO: 2. дальше закидываем запрос в LLM
    extra_body = {"prompt_logprobs": config.count_logprobs}
    extra_body.update(config.extra_body)

    response = await client.chat.completions.create(
        model=config.model,
        messages=messages,
        logprobs=True,
        top_logprobs=config.count_logprobs,  # Берем топ-k вариантов для расчета неопределенности
        extra_body=extra_body,
        **config.params_extra,
    )

    answer = str(response.choices[0].message.content)

    # INFO: 3. Проходим по каждому сгенерированному токену
    ta = TypeAdapter(list[None | dict[str, PromptLogprob]])
    if not response.model_extra:
        raise RuntimeError("Can't compute without prompt logprobs")
    if "prompt_logprobs" not in response.model_extra:
        raise RuntimeError("Can't compute without prompt logprobs")
    prompt_logprobs = ta.validate_python(response.model_extra["prompt_logprobs"])
    return answer, prompt_logprobs
