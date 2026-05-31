import json
import logging
import os
import traceback

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from clearml import Dataset, Task, TaskTypes
from pyarrow import Table

from src.evaluators.cmp_cme.config import AppSettings
from src.schemas import TA_logprob_list
from src.utils.base import (
    configure_logging,
)
from src.utils.startup import init_config

from .save import store_parquet

logger = logging.getLogger(__name__)


async def main():
    c_task: Task = Task.init(
        project_name="RAG_Metrics",
        task_name="Token_Metrics",
        task_type=TaskTypes.data_processing,
        tags=["CrossModel"],
        reuse_last_task_id=False,
    )
    c_task.set_comment("Вычисление метрик по токенам CrossModel")

    config = init_config(conf_type=AppSettings, task=c_task)
    random_state = np.random.RandomState(seed=config.seed)
    configure_logging(config.logging_conf_file)

    # INFO: получаем датасет с вопросами и контекстами
    # и логпробами по конкретной модели
    dataset = Dataset.get(
        dataset_project="RAG_Metrics",
        dataset_name="Build logprobs",
        dataset_tags=[config.llm.model, str(config.llm.count_logprobs)],
        alias="logprobs_dataset",
    )
    dataset_path = dataset.get_local_copy()

    _qa_set_file = os.path.join(dataset_path, "dataset_QA.csv")
    passages_file = os.path.join(dataset_path, "passages.json")
    logprobs_file = os.path.join(dataset_path, "logprobs.parquet")

    table: Table = pq.read_table(
        logprobs_file,
        columns=["eval_id", "label", "prompt_logprob", "prefix_length", "ok"],
    )
    with open(passages_file, "r") as f:
        passages = json.load(f)

    if config.count > 0:
        idx = random_state.choice(table.num_rows, size=config.count, replace=False)
        sample = table.take(
            pa.array(np.sort(idx), type=pa.int64())
        )  # сортировка индексов ускоряет take
        logger.info(
            f"Logprobs file truncated from {table.num_rows} to: {sample.num_rows}"
        )

    filtered_table = table.filter(pc.field("ok") == True)

    passage_ids = filtered_table.column("passage_id")
    questions = filtered_table.column("question")
    answers = filtered_table.column("answer")
    prompt_logprobs = filtered_table.column("prompt_logprob")
    prefix_lengths = filtered_table.column("prefix_length")

    for passage_id, question, answer, prompt_logprob, prefix_length in zip(
        passage_ids, questions, answers, prompt_logprobs, prefix_lengths
    ):
        passage_id: str = str(passage_id)
        question: str = str(question)
        answer: str = str(answer)
        prefix_length: int = int(prefix_length)
        TA_logprob_list.validate_json(prompt_logprob)

    results: list[ExperimentResult] = []

    out_folder, df = store_parquet(results=results)

    logger_c = c_task.get_logger()
    df_ok = df[df["ok"]]
    logger_c.report_single_value("ok_rows", df_ok.shape[0])

    new_dataset = Dataset.create(
        dataset_name="MuSeRC_QA_logprobs",
        dataset_project="RAG_Metrics",
        dataset_tags=["parquet", config.llm.model, str(config.llm.count_logprobs)],
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
    asyncio.run(main())
