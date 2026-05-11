from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.collections import Faithfulness

sample = SingleTurnSample(
    user_input="When was the first super bowl?",
    response="The first superbowl was held on Jan 15, 1967",
    retrieved_contexts=[
        "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
    ],
)
scorer = Faithfulness(llm=evaluator_llm)
await scorer.single_turn_ascore(sample)


import asyncio
import logging
import os
import traceback

import numpy as np
import pandas as pd
from clearml import Dataset, Task, TaskTypes
from openai import AsyncOpenAI
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from src.evaluators.schemas import EvalIn, EvalOut
from src.evaluators.similarity.config import AppSettings
from src.utils.base import (
    calculate_similarity,
    configure_logging,
    create_openai_client,
)
from src.utils.report import log_bin_report
from src.utils.startup import init_config
from src.utils.visual import plot_pr_binary, plot_roc_auc_binary, plot_true_lie_distrib

logger = logging.getLogger(__name__)


async def main():
    c_task: Task = Task.init(
        project_name="RAG_Metrics",
        task_name="RAGAS evaluation",
        task_type=TaskTypes.testing,
        tags=["eval", "RAGAS"],
        reuse_last_task_id=False,
    )
    c_task.set_comment("Мета оценка метрики семантического сходства")

    config = init_config(conf_type=AppSettings, task=c_task)
    random_state = np.random.RandomState(seed=config.seed)
    configure_logging(config.logging_conf_file)

    client = create_openai_client(config=config.embed)
    semaphore = asyncio.Semaphore(config.embed.async_cals)

    # получаем датасет
    dataset = Dataset.get(
        dataset_project="RAG_Metrics",
        dataset_name="MuSeRC_QA_eval",
        dataset_version="1.0.0",
        alias="eval_dataset",
    )
    dataset_path = dataset.get_local_copy()

    qa_set_file = os.path.join(dataset_path, "dataset_QA.csv")
    qa_dataset = pd.read_csv(qa_set_file, index_col=0)
    qa_dataset.index = qa_dataset.index.astype(int)
    logger.info(f"QA readed from {qa_set_file} shape: {qa_dataset.shape}")
    if config.count > 0:
        qa_dataset = qa_dataset.sample(n=config.count, random_state=random_state)

    results: list[EvalOut] = []
    counter = 0

    qa_result = pd.DataFrame.from_records(data=results, index="eval_id")
    logger_c = c_task.get_logger()

    y_true = qa_result["label"].values
    y_score = qa_result["eval_score"].values

    if not isinstance(y_true, np.ndarray):
        raise
    if not isinstance(y_score, np.ndarray):
        raise

    # INFO: 0. вычисляем лучший трэшхолд
    # 1. Считаем FPR, TPR и пороги
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # 2. Вычисляем Youden's J statistic для каждого порога
    j_scores = tpr - fpr

    # 3. Находим индекс максимального значения
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    logger.info(f"Лучший порог по ROC (Youden's J): {best_threshold:.4f}")
    logger_c.report_single_value(
        name="best_roc_threshold", value=round(best_threshold, 4)
    )

    y_pred = (y_score >= best_threshold).astype(int)

    # INFO: 1. строим классификационный отчёт
    target_names = ["Ложь", "Истина"]
    test_report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
    )
    if isinstance(test_report, str):
        logger.info("\n" + test_report)

    test_report_d = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
    )
    if not isinstance(test_report_d, dict):
        raise

    report_d = pd.DataFrame.from_dict(test_report_d)

    logger_c.report_table(
        title="Eval Report",
        series="classification report",
        table_plot=report_d.T.round(2),
    )

    # INFO: 2. Логируем метрики по классам
    log_bin_report(
        logger_c=logger_c, test_report_d=test_report_d, target_names=target_names
    )

    # INFO: 3. Логируем PR-ROC кривые
    fig_pr, ap = plot_pr_binary(
        y_true=y_true,
        y_score=y_score,
        class_name="Истина",
    )
    logger_c.report_plotly(
        title="Eval Report",
        series="PR кривая",
        figure=fig_pr,
    )
    logger_c.report_single_value("AP", ap)
    fig_roc, auc = plot_roc_auc_binary(
        y_true=y_true,
        y_score=y_score,
        class_name="Истина",
    )
    logger_c.report_single_value("AUC", auc)
    logger_c.report_plotly(
        title="Eval Report",
        series="ROC кривая",
        figure=fig_roc,
    )

    # INFO: 4. Логируем confusionMatrix
    # 2. Переводим вероятности в бинарные предсказания (0 или 1) по порогу 0.5

    # 3. Вычисляем саму матрицу ошибок
    cm = confusion_matrix(y_true, y_pred)

    # 4. Логируем готовую матрицу в ClearML
    logger_c.report_confusion_matrix(
        title="Eval Report",
        series="Матрица Коллизий",  # Название графика
        matrix=cm,  # Передаем посчитанную матрицу
        xaxis="Прогноз",
        yaxis="Реальность",
        xlabels=target_names,
        ylabels=target_names,
        extra_layout={
            "texttemplate": "%{z}",
            "colorscale": [
                [0.00, "white"],
                [0.40, "white"],  # до ~65% диапазона чисел — почти белый
                [0.65, "rgb(220,235,250)"],
                [0.75, "rgb(140,185,225)"],
                [1.00, "rgb(40,100,170)"],
            ],
            "textfont": {"size": 24},
            "font": {"size": 16},
        },
    )

    # INFO: 5. Распределение Реально правильных и неправильных
    # с трэшхолдом
    # Create distplot with curve_type set to 'normal'
    fig_dist = plot_true_lie_distrib(
        y_true=y_true,
        y_score=y_score,
        eval_ids=qa_result.index.values,
        target_names=["Истина", "Ложь"],
    )
    logger_c.report_plotly(
        title="SS Report",
        series="Распределение близости ответов",
        figure=fig_dist,
    )

    _ = c_task.flush(wait_for_uploads=True)
    _ = c_task.mark_completed(status_message="completed")
    c_task.close()


if __name__ == "__main__":
    asyncio.run(main())
