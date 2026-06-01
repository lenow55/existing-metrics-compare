import numpy as np
from openai import OpenAI
from scipy.special import logsumexp

# 1. Инициализируем клиент (укажите адрес вашего запущенного vLLM сервера)
client = OpenAI(
    base_url="https://dev02-lb.gpt.dks.lanit.ru/v1",
    api_key="sk-6910b62a54e611f1a3e4bab6455e3028",  # vLLM обычно не требует ключ, можно ввести любую строку
)

model_name = "local/google/gemma-4-31B-it"


def calculate_normalized_entropy(top_logprobs_list) -> float:
    """
    Вычисляет НОРМАЛИЗОВАННУЮ энтропию Шеннона (от 0.0 до 1.0)
    на основе списка логпробов.
    """
    K = len(top_logprobs_list)

    # Если альтернатив нет или всего одна, неопределенность равна 0
    if K <= 1:
        return 0.0

    raw_logprobs = np.array(
        [alt.logprob for alt in top_logprobs_list], dtype=np.float64
    )

    # Стабильная нормализация логпробов через LSE
    lse_constant = logsumexp(raw_logprobs)
    normalized_logprobs = raw_logprobs - lse_constant
    normalized_probs = np.exp(normalized_logprobs)

    # Исходная энтропия в натах
    entropy = -np.sum(normalized_probs * normalized_logprobs)

    # Вычисляем теоретический максимум для данного K
    max_entropy = np.log(K)

    # Нормализуем (ограничиваем сверху 1.0 на случай микроошибок float)
    normalized_entropy = entropy / max_entropy

    return float(normalized_entropy)


# 2. Отправляем запрос с флагами logprobs и top_logprobs
response = client.chat.completions.create(
    model=model_name,
    messages=[{"role": "user", "content": "Напиши историю"}],
    max_tokens=300,
    temperature=1.0,
    presence_penalty=2.0,  # Жестко штрафуем за появление новых токенов
    frequency_penalty=2.0,  # Жестко штрафуем за частоту
    logprobs=True,
    top_logprobs=20,
)

generated_tokens_logprobs = response.choices[0].logprobs.content

print(
    f"{'Токен':<15} | {'Logprob выбранного':<18} | {'Макс. Logprob в топ-20':<22} | {'Истинное совпадение?'}"
)
print("-" * 80)

# Предположим, вы получили ответ `generated_tokens_logprobs` от сервера vLLM:
print(f"{'Токен':<15} | {'Logprob':<10} | {'Энтропия (Уверенность модели)':<30}")
print("-" * 65)

for token_info in generated_tokens_logprobs:
    chosen_token = token_info.token
    chosen_logprob = token_info.logprob

    # Передаем список топ-20 альтернатив в функцию
    token_entropy = calculate_normalized_entropy(token_info.top_logprobs)

    # Интерпретация:
    # Маленькая энтропия (~0.0) -> У модели нет сомнений, выбор предопределен.
    # Высокая энтропия (> 1.5) -> В топ-20 много сильных альтернативных вариантов, модель «колеблется».
    print(
        f"{repr(chosen_token):<15} | {chosen_logprob:<10.4f} | {token_entropy:<30.4f}"
    )
