from typing import NotRequired, TypedDict

from pydantic import BaseModel, Field, TypeAdapter


# INFO: Контейнеры для хранения логпробов
class TokenImportance(TypedDict):
    token: str
    importance: float


TA_tokens_list = TypeAdapter(list[TokenImportance])


class TokenEntropy(TypedDict):
    token: str
    entropy: float


class WordInfo(TypedDict):
    word: str
    start: int
    end: int


class WordInfoRes(WordInfo):
    entropy: float


class WordImportance(WordInfo):
    importance: float


TA_words_list = TypeAdapter(list[WordImportance])


# INFO: Контейнеры для хранения сценариев запросов
class Scenario(TypedDict):
    text: str
    name: str
    label: NotRequired[int]


class ScenarioResult(Scenario):
    logprobs: list[TokenEntropy]


class PtbScenario(TypedDict):
    context: str
    name: str
    question: str
    reference: str
    label: NotRequired[int]


class PtbScenarioRes(PtbScenario):
    logprobs: list[TokenEntropy]
    words: list[WordInfoRes]
    answer: str


# INFO: Контейнеры для валидации вывода VLLM


class PromptLogprob(BaseModel):
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


# INFO: контейнер для подготовленных данных
class Check(TypedDict):
    question: str
    answer: str
    passage_id: str


# INFO: контейнер для запроса в ллм
class CheckLlmIn(Check):
    check_id: int


TA_logprob_list = TypeAdapter(list[None | dict[str, PromptLogprob]])


class CheckStage1Out(CheckLlmIn):
    gen_answer: str
    prompt_logprobs: list[None | dict[str, PromptLogprob]]
    similarity: float


class Stage3Out(CheckLlmIn):
    ptb_words: list[WordImportance]
    passage: str


class Stage3In(CheckLlmIn):
    passage: str
    ptb_words: str


class Stage4Out(CheckStage1Out):
    passage: str
    ptb_words: str


class Stage2Out(TypedDict):
    check_id: int
    tokens_importances: list[TokenImportance]
    words_importances: list[WordImportance]
