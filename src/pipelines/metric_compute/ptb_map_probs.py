import argparse
import json
import logging
import os

import pyarrow.compute as pc
import pyarrow.parquet as pq
from clearml import Dataset, Task, TaskTypes
from pyarrow import Table
from pydantic import Field
from pydantic_settings import BaseSettings

from src.config import ChatLLMConfig
from src.metrics import map_logprobs2parts
from src.schemas import (
    LogprobParts,
    TA_ans_logprob_list,
    TA_logprob_list,
)
from src.utils.base import (
    configure_logging,
)

from .save import ExperimentResult, store_parquet

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
        task_name="Logprobs_Mapping",
        task_type=TaskTypes.data_processing,
        reuse_last_task_id=False,
    )
    c_task.set_comment(
        "Мапинг логпробов на разные части входного текста. (Пертурбированный) Подготовка к метрикам"
    )

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

    dataset_path = dataset.get_local_copy()
    _qa_set_file = os.path.join(dataset_path, "dataset_QA.csv")
    _passages_file = os.path.join(dataset_path, "passages.json")
    logprobs_file = os.path.join(dataset_path, "logprobs.parquet")

    configure_logging(config.logging_conf_file)

    table: Table = pq.read_table(logprobs_file)

    filtered_table = table.filter(pc.field("ok") == True)

    eval_ids = filtered_table.column("eval_id")
    passage_ids = filtered_table.column("passage_id")
    questions = filtered_table.column("question")
    answers = filtered_table.column("answer")
    prompt_logprobs = filtered_table.column("prompt_logprob")
    prefix_lengths = filtered_table.column("prefix_length")
    try:
        top_logprobs = filtered_table.column("top_logprob")
    except KeyError:
        top_logprobs = [None] * table.num_rows

    results: list[ExperimentResult] = []
    for (
        eval_id,
        passage_id,
        question,
        answer,
        prompt_logprob,
        prefix_length,
        top_logprob,
    ) in zip(
        eval_ids,
        passage_ids,
        questions,
        answers,
        prompt_logprobs,
        prefix_lengths,
        top_logprobs,
    ):
        eval_id: int = int(eval_id)
        passage_id: str = str(passage_id)
        question: str = str(question)
        answer: str = str(answer)
        prefix_length: int = int(prefix_length)
        prompt_logprob = TA_logprob_list.validate_json(prompt_logprob.as_py())

        if not top_logprob:
            top_logprob = []
        else:
            top_logprob = TA_ans_logprob_list.validate_json(top_logprob.as_py())

        try:
            logprob_parts: LogprobParts = map_logprobs2parts(
                prompt_logprob=prompt_logprob,
                top_logprob=top_logprob,
                question=question,
                prefix_length=prefix_length,
            )
        except RuntimeError:
            logger.warning(f"Bad question in table {eval_id}")
            continue
        result = ExperimentResult(
            eval_id=eval_id,
            instruct=logprob_parts.instruct,
            context=logprob_parts.context,
            question=logprob_parts.question,
            answer=logprob_parts.answer,
        )
        results.append(result)

    out_folder, _ = store_parquet(results=results)

    new_dataset = Dataset.create(
        dataset_project="RAG_Metrics",
        dataset_name="Logprobs_Mapping",
        dataset_tags=dataset.tags,
        use_current_task=True,
        description="Датасет с сгенерированными логпробами",
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
