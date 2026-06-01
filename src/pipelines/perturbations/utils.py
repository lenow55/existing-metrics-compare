import re

import numpy as np
from transformers import Pipeline


def replace_masks_with_inverse_probability(
    text: str, classifier: Pipeline, rng: np.random.Generator, top_k: int = 5
) -> tuple[str, list[str]]:
    """
    Разбивает текст на предложения по точкам, объединяет их в фрагменты
    весом менее 300 слов, делает прогнозы для масок внутри фрагментов
    и собирает текст обратно.

    :param text: Полный исходный текст любой длины
    :param classifier: Предобученный объект transformers.pipeline("fill-mask")
    :param rng: Инициализированный генератор случайных чисел NumPy
    :param top_k: Количество токенов, запрашиваемых у модели для выбора
    :return: (полный_изменённый_текст, список_вставленных_токенов)
    """

    # === ШАГ 1: Разбивка текста на предложения по точкам ===
    # Используем lookbehind (?<=\.), чтобы разрезать строку строго ПОСЛЕ точки,
    # сохраняя саму точку и последующие пробелы внутри предложений.
    sentences = [s for s in re.split(r"(?<=\.)", text) if s]

    if not sentences:
        return text, []

    # === ШАГ 2: Группировка соседних предложений (лимит < 300 слов) ===
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())

        # Если текущее предложение вместе с накопленными превышает 300 слов,
        # сохраняем накопленный чанк и начинаем новый
        if current_word_count + sentence_word_count >= 200 and current_chunk:
            chunks.append("".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = sentence_word_count
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_word_count

    if current_chunk:
        chunks.append("".join(current_chunk))

    # === ШАГ 3: Пофрагментная обработка и прогнозирование ===
    all_chosen_tokens = []
    final_chunks = []

    for chunk in chunks:
        mask_count = chunk.count("<mask>")

        # Оптимизация: если в этом фрагменте нет масок, сразу отправляем его в финал
        if mask_count == 0:
            final_chunks.append(chunk)
            continue

        # Запускаем модель только для фрагмента с масками
        raw_results = classifier(chunk, top_k=top_k)

        # Стандартизируем вывод под формат списка списков
        if isinstance(raw_results, dict):
            raw_results = [[raw_results]]
        elif (
            isinstance(raw_results, list)
            and len(raw_results) > 0
            and "score" in raw_results[0]
        ):
            raw_results = [raw_results]

        chunk_chosen_tokens = []

        # Расчет инвертированных вероятностей для масок внутри текущего фрагмента
        for mask_predictions in raw_results:
            valid_preds = [p for p in mask_predictions if p["token_str"].strip()]
            if not valid_preds:
                valid_preds = mask_predictions[:1]

            tokens = [pred["token_str"] for pred in valid_preds]
            scores = np.array([pred["score"] for pred in valid_preds])

            inverse_scores = 1.0 - scores
            sum_inverse = np.sum(inverse_scores)

            new_probabilities = (
                inverse_scores / sum_inverse
                if sum_inverse > 0
                else np.ones_like(scores) / len(scores)
            )

            chosen_token = rng.choice(tokens, p=new_probabilities).strip()
            chunk_chosen_tokens.append(chosen_token)
            all_chosen_tokens.append(chosen_token)

        # Точечно заменяем маски в текущем фрагменте
        mask_pattern = re.compile(r"(<mask>)")
        parts = mask_pattern.split(chunk)

        token_idx = 0
        processed_chunk_parts = []

        for part in parts:
            if part == "<mask>":
                if token_idx < len(chunk_chosen_tokens):
                    processed_chunk_parts.append(chunk_chosen_tokens[token_idx])
                    token_idx += 1
                else:
                    processed_chunk_parts.append(part)
            else:
                processed_chunk_parts.append(part)

        final_chunks.append("".join(processed_chunk_parts))

    # === ШАГ 4: Сборка исходного текста из обработанных фрагментов ===
    processed_text = "".join(final_chunks)

    return processed_text, all_chosen_tokens
