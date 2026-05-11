from datetime import datetime
from typing import Any, ClassVar, NotRequired, TypedDict
from pydantic import Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class LLMParams(TypedDict):
    temperature: NotRequired[float]
    frequency_penalty: NotRequired[float]
    seed: NotRequired[int]
    stop: NotRequired[str]
    presence_penalty: NotRequired[float]
    top_p: NotRequired[float]
    max_tokens: NotRequired[int]
    max_retries: NotRequired[int]


class EmbedParams(TypedDict):
    dimensions: NotRequired[int]
    user: NotRequired[str]


class LLMConfig(BaseSettings):
    model: str
    api_key: SecretStr = SecretStr(secret_value="EMPTY")
    base_url: str
    timeout: int = 50
    async_cals: int = 5
    proxy_url: str | None = None
    extra_body: dict[str, Any] = {}


class ChatLLMConfig(LLMConfig):
    count_logprobs: int = 5
    params_extra: LLMParams = LLMParams()


class EmbedLLMConfig(LLMConfig):
    params_extra: EmbedParams = EmbedParams()


class AppSettings(BaseSettings):
    logging_conf_file: str

    llm: ChatLLMConfig
    embed: EmbedLLMConfig | None = (
        None  # Опциональная конфигурация для модели эмбединга
    )

    entropy_threshold: float = Field(
        default=0.8, description="Порог, выше которого токен считается 'неуверенным'"
    )
    max_entropy_scale: float = Field(
        default=2.5, description="Значение энтропии для максимальной насыщенности цвета"
    )

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        json_file=(
            "config.json",
            "config_debug.json",
        )
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            JsonConfigSettingsSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )


class MetadataFileInfo(AppSettings):
    date: datetime = datetime.now()
