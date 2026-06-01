import re

import numpy as np
from transformers import Pipeline


def crop_context_around_masks(text: str, window_words: int = 150) -> str:
    """
    Обрезает текст, оставляя вокруг масок <mask> безопасное окно из слов.
    """
    words = text.split()

    # Ищем точное попадание тега <mask> внутрь слова
    mask_indices = [i for i, word in enumerate(words) if "<mask>" in word]

    if not mask_indices:
        return text  # Если масок почему-то нет, возвращаем текст целиком

    start_idx = max(0, mask_indices[0] - window_words)
    end_idx = min(len(words), mask_indices[-1] + window_words)

    return " ".join(words[start_idx:end_idx])


def replace_masks_with_inverse_probability(
    text: str, classifier: Pipeline, rng: np.random.Generator, top_k: int = 5
) -> tuple[str, list[str]]:
    """
    Заменяет маски <mask> токенами с инвертированной вероятностью.
    Идеально сохраняет все внешние пробелы в тексте.
    """
    # 1. Получаем предсказания модели
    raw_results = classifier(text, top_k=top_k)

    # Приводим к единому формату (список списков)
    if isinstance(raw_results, dict):
        raw_results = [[raw_results]]
    elif (
        isinstance(raw_results, list)
        and len(raw_results) > 0
        and "score" in raw_results[0]
    ):
        raw_results = [raw_results]

    chosen_tokens = []

    # 2. Расчет инвертированных вероятностей для каждой маски
    for mask_predictions in raw_results:
        # Фильтруем пустые строки/токены (бывает на дне топа при больших top_k)
        valid_preds = [p for p in mask_predictions if p["token_str"].strip()]
        if not valid_preds:
            valid_preds = mask_predictions[:1]

        tokens = [pred["token_str"] for pred in valid_preds]
        scores = np.array([pred["score"] for pred in valid_preds])

        # Инвертируем и нормализуем
        inverse_scores = 1.0 - scores
        sum_inverse = np.sum(inverse_scores)

        new_probabilities = (
            inverse_scores / sum_inverse
            if sum_inverse > 0
            else np.ones_like(scores) / len(scores)
        )

        # Делаем случайный выбор токена
        chosen_token = rng.choice(tokens, p=new_probabilities)
        chosen_tokens.append(chosen_token)

    # 3. Сборка текста: режем строго по вашему тегу <mask>
    mask_pattern = re.compile(r"(<mask>)")
    parts = mask_pattern.split(text)

    token_idx = 0
    final_parts = []

    for part in parts:
        if part == "<mask>":
            if token_idx < len(chosen_tokens):
                # .strip() убирает внутренние технические пробелы BPE-токенизатора
                final_parts.append(chosen_tokens[token_idx].strip())
                token_idx += 1
            else:
                final_parts.append(part)
        else:
            # Возвращаем куски оригинального текста со ВСЕМИ его пробелами
            final_parts.append(part)

    return "".join(final_parts), chosen_tokens
