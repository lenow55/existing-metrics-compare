import logging
import math
import os

import numpy as np
import pandas as pd
import plotly.express as px
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from clearml import Dataset, Task, TaskTypes
from joblib import Parallel, delayed
from pyarrow import Table
from scipy.stats import mannwhitneyu

from src.schemas import PromptLogprob, TA_logprob_list
from src.utils.base import (
    configure_logging,
)
from src.utils.report import classifier_report_plan
from src.utils.startup import init_config

from .config import AppSettings

logger = logging.getLogger(__name__)


# INFO: PPL = exp(-1/N*sum(logprob(t_n)))
# Сумма первых логпробов поделённая на -N (длина последовательности)
# и взятая экспонента

CHUNK = 1000


def compute_entropy(
    logprobs: list[None | dict[str, PromptLogprob]],
    prefix_length: int,
):
    """
    Вычисляет Cross-Model Entropy (CME) на основе топ-N логпроб от верификатора.

    Для каждого шага генерации топ-N логпробы перенормируются в распределение
    вероятностей, после чего рассчитывается локальная энтропия Шеннона.
    Итоговый результат — среднее арифметическое энтропий по всей последовательности.
    """
    # Отрезаем промт, оставляем только сгенерированный ответ
    ans_logprobs = logprobs[prefix_length:]

    step_entropies: list[float] = []

    for lp_dict in ans_logprobs:
        if not isinstance(lp_dict, dict) or not lp_dict:
            continue

        # Собираем все доступные логпробы для текущего шага (топ-N)
        # Поддерживается структура как vLLM (значение — объект/словарь с ключом 'logprob'),
        # так и плоский словарь {token: logprob}
        current_step_logprobs: list[float] = []
        for val in lp_dict.values():
            current_step_logprobs.append(val["logprob"])

        if not current_step_logprobs:
            continue

        # 1. Переводим логпробы в реальные вероятности: p = exp(logprob)
        raw_probs = [math.exp(lp) for lp in current_step_logprobs]

        # 2. Нормализуем вероятности, так как у нас на руках только топ-N токенов,
        # а для корректной энтропии их сумма на шаге должна быть равна 1.0
        sum_raw_probs = math.fsum(raw_probs)
        if sum_raw_probs <= 0:
            continue

        norm_probs = [p / sum_raw_probs for p in raw_probs]

        # 3. Считаем энтропию Шеннона для текущего шага: H = -sum(p * log(p))
        # Используем norm_probs для расчета весов и логарифмов
        step_entropy = 0.0
        for p in norm_probs:
            if p > 0:
                step_entropy -= p * math.log(p)

        step_entropies.append(step_entropy)

    # Если валидных шагов не обнаружено
    if not step_entropies:
        return float("inf")

    # 4. Возвращаем среднюю энтропию по всей последовательности ответов
    return math.fsum(step_entropies) / len(step_entropies)


def _score_batch(items: list[tuple[str, int]]) -> list[float]:
    return [compute_entropy(TA_logprob_list.validate_json(s), p) for s, p in items]


def main():
    c_task: Task = Task.init(
        project_name="RAG_Metrics",
        task_name="CM Entropy evaluation",
        task_type=TaskTypes.testing,
        tags=["eval", "Entropy", "CrossModel"],
        reuse_last_task_id=False,
    )
    c_task.set_comment("Вычисление Entropy на модели CrossModel")

    config = init_config(conf_type=AppSettings, task=c_task)
    random_state = np.random.RandomState(seed=config.seed)
    configure_logging(config.logging_conf_file)

    # получаем датасет
    dataset = Dataset.get(
        dataset_project="RAG_Metrics",
        dataset_name="Build logprobs",
        dataset_tags=[config.llm.model],
        alias="eval_dataset",
    )
    dataset_path = dataset.get_local_copy()

    qa_set_file = os.path.join(dataset_path, "dataset_QA.csv")
    _ = os.path.join(dataset_path, "passages.json")
    logprobs_file = os.path.join(dataset_path, "logprobs.parquet")

    qa_dataset = pd.read_csv(qa_set_file, index_col=0)
    qa_dataset.index = qa_dataset.index.astype(int)
    logger.info(f"QA readed from {qa_set_file} shape: {qa_dataset.shape}")

    table: Table = pq.read_table(
        logprobs_file,
        columns=["eval_id", "label", "prompt_logprob", "prefix_length", "ok"],
    )
    logger.info(f"QA readed from {logprobs_file} shape: {table.shape}")

    if config.count > 0:
        idx = random_state.choice(table.num_rows, size=config.count, replace=False)
        sample = table.take(
            pa.array(np.sort(idx), type=pa.int64())
        )  # сортировка индексов ускоряет take
        logger.info(
            f"Logprobs file truncated from {table.num_rows} to: {sample.num_rows}"
        )

    filtered_table = table.filter(pc.field("ok") == True)

    prompt_logprobs = filtered_table.column("prompt_logprob").to_pylist()
    prefix_lengths = filtered_table.column("prefix_length").to_pylist()

    pairs = list(zip(prompt_logprobs, prefix_lengths))
    chunks = [pairs[i : i + CHUNK] for i in range(0, len(pairs), CHUNK)]
    results_nested = Parallel(n_jobs=-1, backend="loky")(
        delayed(_score_batch)(c) for c in chunks
    )
    scores = [x for sub in results_nested for x in sub]  # pyright: ignore[reportOptionalIterable]
    score_array = pa.array(scores, type=pa.float64())
    filtered_table = filtered_table.append_column("score", score_array)

    qa_result = filtered_table.select(["eval_id", "label", "score"]).to_pandas()
    c_task.register_artifact(name="evaluation_result", artifact=qa_result)
    _ = c_task.flush(wait_for_uploads=True)

    logger_c = c_task.get_logger()
    # Проверим на nan
    logger_c.report_single_value("ok_rows", qa_result.shape[0])

    y_true = qa_result["label"].values
    y_score = qa_result["score"].values

    if not isinstance(y_true, np.ndarray):
        raise
    if not isinstance(y_score, np.ndarray):
        raise

    fig = px.histogram(
        y_score,
        nbins=50,
        marginal="box",
    )
    logger_c.report_plotly(
        title="ScoreDist Report",
        series="Entropy",
        figure=fig,
    )

    # INFO: энтропию тоже надо инвертировать,
    # больше энтропия - хуже
    invert_score = 1 - y_score

    fig = px.histogram(
        invert_score,
        nbins=50,
        marginal="box",
    )
    logger_c.report_plotly(
        title="ScoreDist Report",
        series="Entropy inverted",
        figure=fig,
    )
    _, p_value_mw = mannwhitneyu(
        y_score[y_true == 1],
        y_score[y_true == 0],
        alternative="less",
    )
    logger_c.report_single_value("MWU p-value", float(p_value_mw))

    for lbl in (0, 1):
        s = y_score[y_true == lbl]
        print(f"  label={lbl}: n={len(s)}, median Entropy={np.median(s):.2f}")

    classifier_report_plan(
        y_true=y_true,
        y_score=invert_score,
        index=qa_result.index.values,
        logger_c=logger_c,
        metric_name="Entropy (CMP)",
        show_hist=False,
    )
    _ = c_task.flush(wait_for_uploads=True)
    _ = c_task.mark_completed(status_message="completed")
    c_task.close()


if __name__ == "__main__":
    main()
