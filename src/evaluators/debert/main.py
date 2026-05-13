import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from clearml import Dataset, Task, TaskTypes
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DebertaV2ForSequenceClassification,
    DebertaV2Tokenizer,
)

from src.utils.base import configure_logging
from src.utils.report import classifier_report_plan
from src.utils.startup import init_config

from .config import AppSettings

logger = logging.getLogger(__name__)

MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"


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


def compute_nli_score(
    premise: str,
    hypothesis: str,
    tokenizer: DebertaV2Tokenizer,
    model: DebertaV2ForSequenceClassification,
    device: torch.device,
) -> float:
    """Вычисляет NLI entailment score (0..1) для пары premise/hypothesis."""
    inputs = tokenizer(
        premise,
        hypothesis,
        truncation="only_first",
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0]
    probabilities = torch.softmax(logits, dim=-1)

    # Находим индекс класса 'entailment' из конфигурации модели
    if not isinstance(model.config.id2label, dict):
        raise
    entailment_idx = None
    for k, v in model.config.id2label.items():
        entailment_idx = k
        if v == "entailment":
            break
    if not isinstance(entailment_idx, int):
        raise
    return float(probabilities[entailment_idx].item())


def evaluate_dataset(
    qa_dataset: pd.DataFrame,
    passages: dict[str, str],
    tokenizer: DebertaV2Tokenizer,
    model: DebertaV2ForSequenceClassification,
    device: torch.device,
) -> pd.DataFrame:
    """Вычисляет NLI faithfulness для каждой строки датасета."""
    results: list[ExperimentResult] = []

    for idx, row in qa_dataset.iterrows():
        if not isinstance(idx, int):
            raise RuntimeError("Bad index type")
        try:
            passage = passages[str(int(row["passage_id"]))]
            # premise = контекст, hypothesis = ответ модели
            score = compute_nli_score(
                premise=passage,
                hypothesis=row["response"],
                tokenizer=tokenizer,
                model=model,
                device=device,
            )
            results.append(
                ExperimentResult(
                    id=int(idx),
                    ok=True,
                    faithfulness=score,
                    error=None,
                    label=int(row["label"]),
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("NLI scoring failed for row %s", idx)
            results.append(
                ExperimentResult(
                    id=int(idx),
                    ok=False,
                    faithfulness=None,
                    error=f"{type(exc).__name__}: {exc}",
                    label=int(row["label"]),
                )
            )

    return pd.DataFrame([r.model_dump() for r in results])


def main():
    c_task: Task = Task.init(
        project_name="RAG_Metrics",
        task_name="DeBERTa NLI evaluation Faithfulness",
        task_type=TaskTypes.testing,
        tags=["eval", "DeBERTa", "NLI", "Faithfulness"],
        reuse_last_task_id=False,
    )
    c_task.set_comment("Мета оценка метрики Faithfulness через mDeBERTa NLI")

    config = init_config(conf_type=AppSettings, task=c_task)
    random_state = np.random.RandomState(seed=config.seed)
    configure_logging(config.logging_conf_file)

    # Инициализация модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Загрузка модели %s на %s...", MODEL_NAME, device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # Получаем датасет
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

    # Вычисляем метрику
    qa_result = evaluate_dataset(
        qa_dataset=eval_df,
        passages=passages,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
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
    # Проверим на nan
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
        metric_name="Faithfulness (DeBERTa NLI)",
        show_hist=False,
    )

    _ = c_task.flush(wait_for_uploads=True)
    _ = c_task.mark_completed(status_message="completed")
    c_task.close()


if __name__ == "__main__":
    main()
