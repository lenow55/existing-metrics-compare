import asyncio
import json
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
from ragas.metrics.collections import Faithfulness
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from src.utils.base import (
    configure_logging,
    create_openai_client,
)
from src.utils.report import classifier_report_plan, log_bin_report
from src.utils.startup import init_config
from src.utils.visual import plot_pr_binary, plot_roc_auc_binary, plot_true_lie_distrib

from .config import AppSettings

logger = logging.getLogger(__name__)

os.environ["RAGAS_DO_NOT_TRACK"] = "true"


# Define experiment result structure
class ExperimentResult(BaseModel):
    id: int
    ok: bool = Field(description="Успешно ли посчитана метрика для строки")
    faithfulness: float | None = Field(
        default=None,
        description="0..1 при успехе; None при ошибке",
    )
    error: str | None = Field(
        default=None, description="Текст исключения, если ok=False"
    )
    label: int


# Create experiment function
@experiment(ExperimentResult)
async def run_evaluation(
    row,
    *,
    llm: InstructorBaseRagasLLM,
    sem: asyncio.Semaphore,
    passages: dict[str, str],
):
    faithfulness_s = Faithfulness(llm=llm)

    async with sem:
        try:
            passage = passages[str(int(row["passage_id"]))]
            faithfulness = await faithfulness_s.ascore(
                user_input=row["user_input"],
                response=row["response"],
                retrieved_contexts=[passage],
            )
            return ExperimentResult(
                id=int(row["id"]),
                ok=True,
                faithfulness=float(faithfulness.value),
                error=None,
                label=int(row["label"]),
            )
        except Exception as exc:  # noqa: BLE001 — намеренно широкий перехват для статистики
            logger.exception("Faithfulness failed for row")
            return ExperimentResult(
                id=int(row["id"]),
                ok=False,
                faithfulness=None,
                error=f"{type(exc).__name__}: {exc}",
                label=int(row["label"]),
            )


async def main():
    c_task: Task = Task.init(
        project_name="RAG_Metrics",
        task_name="RAGAS evaluation Faithfulness",
        task_type=TaskTypes.testing,
        tags=["eval", "RAGAS", "Faithfulness"],
        reuse_last_task_id=False,
    )
    c_task.set_comment("Мета оценка метрики Faithfulness RAGAS")

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
    passages_file = os.path.join(dataset_path, "passages.json")

    with open(passages_file, "r") as f:
        passages = json.load(f)

    qa_dataset = pd.read_csv(qa_set_file, index_col=0)
    qa_dataset.index = qa_dataset.index.astype(int)
    logger.info(f"QA readed from {qa_set_file} shape: {qa_dataset.shape}")

    if config.count > 0:
        qa_dataset = qa_dataset.sample(n=config.count, random_state=random_state)
        logger.info(f"QA truncated from {qa_set_file} shape: {qa_dataset.shape}")

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
        name="muse_faithulness",
        llm=r_client,
        sem=sem,
        passages=passages,
    )
    qa_result = exp.to_pandas()
    c_task.register_artifact(name="evaluation_result", artifact=qa_result)
    _ = c_task.flush(wait_for_uploads=True)

    bad_result = qa_result[~qa_result["ok"]]
    if not bad_result.empty:
        logger.warning(f"Count records with fail: {bad_result.shape[0]}")
        bad_result.to_csv("./bad.csv")
    else:
        logger.info("Metrics computed success")

    logger_c = c_task.get_logger()
    qa_result = qa_result[qa_result["ok"]]
    # INFO: Проверим на nan
    qa_result = qa_result[qa_result["faithfulness"].notna()]
    logger_c.report_single_value("ok_rows", qa_result.shape[0])

    y_true = qa_result["label"].values
    y_score = qa_result["faithfulness"].values

    if not isinstance(y_true, np.ndarray):
        raise
    if not isinstance(y_score, np.ndarray):
        raise

    classifier_report_plan(
        y_true=y_true,
        y_score=y_score,
        index=qa_result.index.values,
        logger_c=logger_c,
        metric_name="Faithfulness",
        show_hist=False,
    )

    _ = c_task.flush(wait_for_uploads=True)
    _ = c_task.mark_completed(status_message="completed")
    c_task.close()


if __name__ == "__main__":
    asyncio.run(main())
