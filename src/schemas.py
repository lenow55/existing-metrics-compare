from dataclasses import dataclass
from typing import NotRequired, TypedDict

from pydantic import BaseModel, Field, TypeAdapter

# INFO: Контейнеры метрики text unit


class TextUnitMetric(TypedDict):
    """
    универсальный контейнер для частички текста,
    котораям может быть как токеном, так и целым словом
    """

    value: float
    """значение метрики"""
    index: int
    """индекс на выходе текстовой единицы"""
    text_unit: str
    """текстовое представление unicode"""
    position: NotRequired[tuple[int, int]]
    """
    (индекс начального символа в исходном тексте, индекс конечного символа в исходном тексте)
    для частей промпта
    """


@dataclass(frozen=True)
class MetricOutput:
    instruct: list[TextUnitMetric]
    context: list[TextUnitMetric]
    question: list[TextUnitMetric]
    answer: list[TextUnitMetric]


# INFO: Контейнеры для валидации вывода VLLM


class PromptLogprob(TypedDict):
    decoded_token: str
    """The token."""

    logprob: float
    """The log probability of this token, if it is within the top 20 most likely
    tokens.

    Otherwise, the value `-9999.0` is used to signify that the token is very
    unlikely.
    """

    rank: int
    """
    Позиция в отранжированном списке токенов
    """


# INFO: Модели валидации датасета MuSeRC


class Answer(BaseModel):
    """
    Модель отдельного варианта ответа.
    """

    idx: int = Field(
        ...,
        description="Уникальный числовой идентификатор варианта ответа (в рамках одного вопроса).",
    )
    text: str = Field(..., description="Текст варианта ответа.")
    label: int = Field(
        ..., description="Метка правильности ответа: 1 — верный ответ, 0 — неверный."
    )


class Question(BaseModel):
    """
    Модель вопроса к тексту с вариантами ответов.
    """

    idx: int = Field(
        ...,
        description="Уникальный числовой идентификатор вопроса (в рамках одного текста).",
    )
    question: str = Field(..., description="Текст вопроса, задаваемого к отрывку.")
    answers: list[Answer] = Field(
        ..., description="Список предложенных вариантов ответов для данного вопроса."
    )


class PassageData(BaseModel):
    """
    Контейнер для текста и связанных с ним вопросов.
    """

    text: str = Field(
        ...,
        description="Основной текст (отрывок) для чтения, содержащий информацию для ответов.",
    )
    questions: list[Question] = Field(
        ..., description="Список вопросов, относящихся к данному тексту."
    )


class ReadingComprehensionItem(BaseModel):
    """
    Корневая модель для одного элемента датасета.
    """

    idx: int = Field(
        ..., description="Глобальный уникальный идентификатор записи в датасете."
    )
    passage: PassageData = Field(
        ..., description="Объект, содержащий текст и структуру вопросов-ответов."
    )


TA_logprob_list = TypeAdapter(list[None | dict[str, PromptLogprob]])
