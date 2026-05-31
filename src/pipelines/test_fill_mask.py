from transformers import pipeline

classifier = pipeline("fill-mask", model="sberbank-ai/ruRoberta-large")

# Вставляем две маски в разные места (без пробелов перед ними)
text = "Наступила<mask>, птицы улетели на<mask>."

result = classifier(text)

# Поскольку масок несколько, pipeline вернет список СПИСКОВ.
# Каждой маске соответствует свой набор вариантов.
for i, mask_predictions in enumerate(result):
    print(f"\nВарианты для Маски №{i + 1}:")
    for prediction in mask_predictions[
        :3
    ]:  # смотрим топ-3 варианта  # pyright: ignore[reportArgumentType]
        print(
            f"  Слово: '{prediction['token_str']:<12}' | Вероятность: {prediction['score']:.4f}"
        )
