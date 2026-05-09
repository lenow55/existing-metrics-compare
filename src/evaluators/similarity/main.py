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
from src.utils.visual import plot_pr_binary, plot_roc_auc_binary

logger = logging.getLogger(__name__)


async def stage_task(
    eval: EvalIn,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    config: AppSettings,
) -> EvalOut:
    async with semaphore:
        _, score = await calculate_similarity(
            idx=str(eval["eval_id"]),
            reference=eval["reference"],
            answer=eval["answer"],
            client=client,
            config=config.embed,
        )
    return EvalOut(
        eval_score=score,
        **eval,
    )


async def main():
    c_task: Task = Task.init(
        project_name="RAG_Metrics",
        task_name="Cosine SemSim evaluation",
        task_type=TaskTypes.testing,
        tags=["eval", "CosSS", "embed"],
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

    tasks: list[asyncio.Task[EvalOut]] = []
    for idx, row in qa_dataset.iterrows():
        if not isinstance(idx, int):
            raise RuntimeError("Bad index type")
        tasks.append(
            asyncio.create_task(
                stage_task(
                    eval={
                        "eval_id": idx,
                        "answer": str(row["answer"]),
                        "passage_id": str(int(row["passage_id"])),
                        "question": str(row["question"]),
                        "reference": str(row["reference"]),
                        "label": int(row["label"]),
                    },
                    client=client,
                    semaphore=semaphore,
                    config=config,
                )
            )
        )

    results: list[EvalOut] = []
    counter = 0
    try:
        for task in asyncio.as_completed(tasks):
            try:
                cluster_prop = await task
                results.append(cluster_prop)
            except Exception as e:
                logger.warning(
                    f"Error when similarity calculation, record will skip: {e}"
                )
                logger.debug(traceback.format_exc())
                continue
            finally:
                counter = counter + 1
                if counter % 100 == 0:
                    logger.info(f"Processed {counter}/{len(tasks)}")
    except asyncio.exceptions.CancelledError:
        logger.error("Получен Ctrl+C, отменяем задачи...")
        for t in tasks:
            _ = t.cancel()
        _ = await asyncio.gather(*tasks, return_exceptions=True)
        logger.error("Все задачи корректно завершены")
        _ = c_task.flush()
        _ = c_task.mark_completed(status_message="stopped")
        c_task.close()
        exit(1)

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
    logger.info(test_report)
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
            "colorscale": "YlOrRd",
            "textfont": {"size": 16},
            "font": {"size": 13},
            "xaxis": {"tickfont": {"size": 11}},
            "yaxis": {"tickfont": {"size": 11}},
        },
    )
    _ = c_task.flush(wait_for_uploads=True)
    _ = c_task.mark_completed(status_message="completed")
    c_task.close()


if __name__ == "__main__":
    asyncio.run(main())
