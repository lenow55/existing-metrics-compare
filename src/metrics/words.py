from src.metrics.base import register
from src.metrics.tokens import step_token_entropy
from src.schemas import LogprobStep, TextUnitMetric


def step_word_entropy(logprobs: list[LogprobStep]) -> list[TextUnitMetric]:
    """
    Рассчитывает среднюю энтропию по словам, используя базовую функцию
    по-токенового расчета step_token_entropy.
    """
    # 1. Получаем энтропию для каждого токена через уже готовую функцию
    token_metrics = step_token_entropy(logprobs)

    result: list[TextUnitMetric] = []
    if not token_metrics:
        return result

    current_word_text = ""
    current_word_entropies: list[float] = []
    word_index = 0

    # Спецсимволы, обозначающие начало нового слова в различных токенизаторах
    word_start_prefixes = (" ", " ", "Ġ", "\n", "\t")

    # 2. Группируем полученные токены в слова
    for tm in token_metrics:
        # Поскольку TextUnitMetric — это TypedDict, обращаемся по ключам
        token_text = tm["text_unit"]
        entropy = tm["value"]

        # Проверяем, начинается ли токен с пробельного символа
        is_new_word = token_text.startswith(word_start_prefixes)

        # Начинаем новое слово, только если текущий буфер не пустой
        # (strip() защищает от разбиения из-за нескольких пробелов подряд или в начале текста)
        if is_new_word and current_word_text.strip():
            # Усредняем энтропию накопленных токенов текущего слова
            avg_entropy = sum(current_word_entropies) / len(current_word_entropies)

            result.append(
                TextUnitMetric(
                    value=avg_entropy,
                    index=word_index,
                    text_unit=current_word_text,
                )
            )
            word_index += 1

            # Сбрасываем буфер для нового слова
            current_word_text = token_text
            current_word_entropies = [entropy]
        else:
            # Продолжаем склеивать части текущего слова
            current_word_text += token_text
            current_word_entropies.append(entropy)

    # 3. Не забываем добавить последнее собранное слово после выхода из цикла
    if current_word_text:
        avg_entropy = sum(current_word_entropies) / len(current_word_entropies)
        result.append(
            TextUnitMetric(
                value=avg_entropy,
                index=word_index,
                text_unit=current_word_text,
            )
        )

    return result


register(id="word_mean_entropy", f_metric=step_word_entropy)
