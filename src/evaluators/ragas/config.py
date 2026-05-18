from pathlib import Path
from typing import ClassVar, NotRequired, TypedDict

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from src.config import LLMConfig, LLMParams

FILE_ROOT = Path(__file__).resolve().parent


class RagasParams(LLMParams):
    max_retries: NotRequired[int]


class ChatLLMConfig(LLMConfig):
    params_extra: RagasParams = RagasParams()


class AppSettings(BaseSettings):
    logging_conf_file: str
    llm: ChatLLMConfig
    count: int = Field(default=-1, description="количество для обработки")
    seed: int = Field(default=42)
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        json_file=(FILE_ROOT / "config.json")
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
