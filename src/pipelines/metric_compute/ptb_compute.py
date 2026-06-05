import argparse
import logging
import os

import pyarrow.parquet as pq
from clearml import Dataset, Task, TaskTypes
from pyarrow import Table
from pydantic import Field
from pydantic_settings import BaseSettings

from src.config import ChatLLMConfig
from src.metrics import METRICS_HUB
from src.schemas import (
    TA_logprob_steps_list,
)
from src.utils.base import (
    configure_logging,
)

from .save_c import ExperimentResult, store_parquet

logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
_ = parser.add_argument(
    "-d",
    "--dataset",
    type=str,
    required=True,
    help="ID датасета с пертурбацией",
)


class AppSettings(BaseSettings):
    logging_conf_file: str
    llm: ChatLLMConfig
    count: int = Field(default=-1, description="количество для обработки")
    seed: int = Field(default=42)


def main(args: argparse.Namespace):
    if not isinstance(args.dataset, str):
        raise RuntimeError("Не указан ID датасета с пертурбациями")

    c_task: Task = Task.init(
        project_name="RAG_Metrics",
        task_name="Logprobs_Scoring",
        task_type=TaskTypes.data_processing,
        reuse_last_task_id=False,
    )
    c_task.set_comment("Вычисление скоров значимости по логпробам")

    dataset = Dataset.get(
        dataset_id=args.dataset,
        alias="logprobs_dataset",
    )
    dataset_task: Task = Task.get_task(task_id=dataset.id)

    task_config = dataset_task.get_parameters_as_dict().get("Hyperparameters")
    if not isinstance(task_config, dict):
        raise ValueError("Can't get configuration object")
    config = AppSettings(**task_config)
    c_task.connect(task_config, name="Hyperparameters")

    task_args = dataset_task.get_parameters_as_dict().get("Args")
    c_task.connect(task_args, name="Args")

    # достаём имя метрики для скоринга
    if not isinstance(task_args, dict):
        raise RuntimeError("Args not valid")

    metric = task_args.get("metric")
    if not isinstance(metric, str):
        raise RuntimeError("Can't get metric name from args")

    if metric not in METRICS_HUB:
        raise RuntimeError(f"Metric {metric} not found in HUB {METRICS_HUB}")

    configure_logging(config.logging_conf_file)

    dataset_path = dataset.get_local_copy()
    _qa_set_file = os.path.join(dataset_path, "dataset_QA.csv")
    _passages_file = os.path.join(dataset_path, "passages.json")
    logprobs_file = os.path.join(dataset_path, "logprobs_parted.parquet")

    table: Table = pq.read_table(logprobs_file)

    eval_ids = table.column("eval_id")
    instruct_p = table.column("instruct")
    context_p = table.column("context")
    question_p = table.column("question")
    answer_p = table.column("answer")

    results: list[ExperimentResult] = []
    target_func = METRICS_HUB[metric]
    logger.info(f"User {metric} function for scoring")
    for (
        eval_id,
        instruct,
        context,
        question,
        answer,
    ) in zip(
        eval_ids,
        instruct_p,
        context_p,
        question_p,
        answer_p,
    ):
        eval_id: int = int(eval_id)
        instruct = TA_logprob_steps_list.validate_json(instruct.as_py())
        context = TA_logprob_steps_list.validate_json(context.as_py())
        question = TA_logprob_steps_list.validate_json(question.as_py())
        answer = TA_logprob_steps_list.validate_json(answer.as_py())

        try:
            instruct_scores = target_func(logprobs=instruct)
            context_scores = target_func(logprobs=context)
            question_scores = target_func(logprobs=question)
            answer_scores = target_func(logprobs=answer)

        except RuntimeError:
            logger.warning(f"Bad scoring in eval_id: {eval_id}")
            continue
        result = ExperimentResult(
            eval_id=eval_id,
            instruct=instruct_scores,
            context=context_scores,
            question=question_scores,
            answer=answer_scores,
        )
        results.append(result)

    out_folder, _ = store_parquet(results=results)

    new_dataset = Dataset.create(
        dataset_project="RAG_Metrics",
        dataset_name="Logprobs_Scoring",
        dataset_tags=dataset.tags,
        use_current_task=True,
        description="Датасет с метриками по логпробам",
        parent_datasets=[dataset],
    )
    _ = new_dataset.add_files(path=out_folder)
    _ = new_dataset.upload()
    _ = new_dataset.finalize()

    _ = c_task.flush(wait_for_uploads=True)
    _ = c_task.mark_completed(status_message="completed")
    c_task.close()


if __name__ == "__main__":
    main(parser.parse_args())
