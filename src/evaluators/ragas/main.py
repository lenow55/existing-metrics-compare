import asyncio
import logging
import os

import numpy as np
import pandas as pd
from clearml import Dataset, Task, TaskTypes
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from ragas import Experiment, experiment
from ragas.backends import InMemoryBackend
from ragas.dataset import Dataset as RDataset
from ragas.llms.base import InstructorBaseRagasLLM, llm_factory
from ragas.metrics.collections import AnswerAccuracy
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from src.utils.base import (
    configure_logging,
    create_openai_client,
)
from src.utils.report import log_bin_report
from src.utils.startup import init_config
from src.utils.visual import plot_pr_binary, plot_roc_auc_binary, plot_true_lie_distrib

from .config import AppSettings

logger = logging.getLogger(__name__)

os.environ["RAGAS_DO_NOT_TRACK"] = "true"


# Define experiment result structure
class ExperimentResult(BaseModel):
    id: int
    ok: bool = Field(description="Успешно ли посчитана метрика для строки")
    answer_accuracy: float | None = Field(
        default=None,
        description="0..1 при успехе; None при ошибке",
    )
    error: str | None = Field(
        default=None, description="Текст исключения, если ok=False"
    )
    label: int


# Create experiment function
@experiment(ExperimentResult)
async def run_evaluation(row, *, llm: InstructorBaseRagasLLM, sem: asyncio.Semaphore):
    answer_accuracy = AnswerAccuracy(llm=llm)

    async with sem:
        try:
            accuracy = await answer_accuracy.ascore(
                user_input=row["user_input"],
                response=row["response"],
                reference=row["reference"],
            )
            return ExperimentResult(
                id=int(row["id"]),
                ok=True,
                answer_accuracy=float(accuracy.value),
                error=None,
                label=int(row["label"]),
            )
        except Exception as exc:  # noqa: BLE001 — намеренно широкий перехват для статистики
            logger.exception("AnswerAccuracy failed for row")
            return ExperimentResult(
                id=int(row["id"]),
                ok=False,
                answer_accuracy=None,
                error=f"{type(exc).__name__}: {exc}",
                label=int(row["label"]),
            )


async def main():
    c_task: Task = Task.init(
        project_name="RAG_Metrics",
        task_name="RAGAS evaluation AnswerAccuracy",
        task_type=TaskTypes.testing,
        tags=["eval", "RAGAS", "AnswerAccuracy"],
        reuse_last_task_id=False,
    )
    c_task.set_comment("Мета оценка метрики семантического сходства")

    config = init_config(conf_type=AppSettings, task=c_task)
    random_state = np.random.RandomState(seed=config.seed)
    configure_logging(config.logging_conf_file)

    client: AsyncOpenAI = create_openai_client(config=config.llm)
    r_client = llm_factory(
        model=config.llm.model,
        provider="openai",
        client=client,
        extra_body=config.llm.extra_body,
        **config.llm.params_extra,
    )
    sem = asyncio.Semaphore(config.llm.async_cals)

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
    eval_df = qa_dataset.rename(
        columns={"question": "user_input", "answer": "response"}
    )
    qa_eval_dataset = RDataset.from_pandas(
        eval_df.reset_index(drop=False),
        "MuSeRC_QA_eval",
        InMemoryBackend(),
    )
    exp: Experiment = await run_evaluation.arun(
        dataset=qa_eval_dataset,
        name="muse_answer_accuracy",
        llm=r_client,
        sem=sem,
    )
    qa_result = exp.to_pandas()
    c_task.register_artifact(name="evaluation_result", artifact=qa_result)

    bad_result = qa_result[~qa_result["ok"]]
    if not bad_result.empty:
        logger.warning(f"Count records with fail: {bad_result.shape[0]}")
        bad_result.to_csv("./bad.csv")
    else:
        logger.info("Metrics computed success")

    logger_c = c_task.get_logger()
    qa_result = qa_result[qa_result["ok"]]
    logger_c.report_single_value("ok_rows", qa_result.shape[0])

    y_true = qa_result["label"].values
    y_score = qa_result["answer_accuracy"].values

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
        show_hist=True,
        bin_size=0.2,
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
