import re

import numpy as np
from transformers import Pipeline


def replace_masks_with_inverse_probability(
    text: str, classifier: Pipeline, rng: np.random.Generator
) -> tuple[str, list[str]]:
    """
    Заменяет маски в тексте токенами, отдавая приоритет МЕНЕЕ вероятным вариантам.

    :param text: Исходный текст с масками (например, "Наступила<mask>, птицы улетели на<mask/>.")
    :param classifier: Предобученный объект transformers.pipeline("fill-mask")
    :param rng: Инициализированный генератор случайных чисел NumPy
    :return: (изменённый_текст, список_вставленных_токенов)
    """
    # Запускаем предсказание через переданный снаружи классификатор
    raw_results = classifier(text)

    # Приводим вывод к единому стандарту (список списков результатов для каждой маски)
    if isinstance(raw_results, dict):
        raw_results = [[raw_results]]
    elif (
        isinstance(raw_results, list)
        and len(raw_results) > 0
        and "score" in raw_results[0]
    ):
        raw_results = [raw_results]  # pyright: ignore[reportGeneralTypeIssues]

    chosen_tokens: list[str] = []

    for mask_predictions in raw_results:  # pyright: ignore[reportGeneralTypeIssues]
        # 1. Извлекаем токены и их исходные вероятности
        tokens: list[str] = [pred["token_str"] for pred in mask_predictions]
        scores = np.array([pred["score"] for pred in mask_predictions])

        # 2. Инвертируем вероятности (плотность сдвигается к низу топа)
        inverse_scores = 1.0 - scores

        # 3. Нормализуем распределение
        sum_inverse = np.sum(inverse_scores)
        if sum_inverse > 0:
            new_probabilities = inverse_scores / sum_inverse
        else:
            new_probabilities = np.ones_like(scores) / len(scores)

        # 4. Делаем случайный выбор токена
        chosen_token = rng.choice(tokens, p=new_probabilities)
        chosen_tokens.append(chosen_token)

    # 5. Заменяем маски в тексте поочередно.
    # Регулярное выражение находит и <mask_и <mask/> в любом регистре
    processed_text: str = text
    mask_pattern = re.compile(r"(?i)<mask/?>")

    for token in chosen_tokens:
        processed_text: str = mask_pattern.sub(token, processed_text, count=1)

    return processed_text, chosen_tokens
