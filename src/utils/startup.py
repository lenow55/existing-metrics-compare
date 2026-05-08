from typing import TypeVar

from clearml import Task
from pydantic_settings import BaseSettings


TConf = TypeVar("TConf", bound=BaseSettings)


def init_config(conf_type: type[TConf], task: Task) -> TConf:
    conf = conf_type()
    # Шаг B: Превращаем Pydantic-модель в словарь
    # Используем model_dump() для Pydantic v2 (или .dict() для v1)
    config_dict = conf.model_dump()

    # Шаг C: Подключаем словарь к ClearML.
    # ВАЖНО: Если задача запущена агентом с новыми параметрами из UI,
    # ClearML автоматически перезапишет значения внутри `config_dict` прямо на этой строке!
    config_dict = task.connect(config_dict, name="Hyperparameters")

    # Шаг D: Пересобираем Pydantic-модель из словаря.
    # Это гарантирует, что если в UI ClearML кто-то введет строку вместо числа,
    # Pydantic выбросит понятную ошибку (ValidationError) до начала тяжелых вычислений.
    final_config = conf_type(**config_dict)  # pyright: ignore[reportAny]

    return final_config
