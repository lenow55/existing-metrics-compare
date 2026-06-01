import argparse
import json
import logging
import os

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from clearml import Dataset, Task, TaskTypes
from pyarrow import Table

from src.evaluators.cmp_cme.config import AppSettings
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
    "-c",
    "--config",
    type=str,
    required=False,
    default="./congig.json",
    help="Путь до файла с конфигурацией",
)
_ = parser.add_argument(
    "-t",
    "--type",
    type=str,
    required=True,
    choices=["NoCTX", "CTX", "PTB"],
    help="Тип прогона: [NoCTX, CTX, PTB]",
)
_ = parser.add_argument(
    "-s",
    "--cross-model",
    action="store_true",
    required=False,
    default=False,
    help="Флаг генерации CrossModel",
)


def main(args: argparse.Namespace):
    if not isinstance(args.config, str):
        raise RuntimeError("Bad argument for config path value")
    if not isinstance(args.type, str):
        raise RuntimeError("Bad argument for type value")
    if not isinstance(args.cross_model, bool):
        raise RuntimeError("Bad argument for cross-model value")

    tags: list[str] = []
    tags.append(args.type)
    if args.cross_model:
        tags.append("CrossModel")

    c_task: Task = Task.init(
        project_name="RAG_Metrics",
        task_name="Logprobs_Mapping",
        task_type=TaskTypes.data_processing,
        tags=tags,
        reuse_last_task_id=False,
    )
    c_task.set_comment(
        "Мапинг логпробов на разные части входного текста. Подготовка к метрикам"
    )
    with open(args.config, "r") as f:
        conf = AppSettings.model_validate_json(f.read())
    config_dict = conf.model_dump(mode="python")
    config_dict = c_task.connect(config_dict, name="Hyperparameters")
    config = AppSettings(**config_dict)  # pyright: ignore[reportAny]

    random_state = np.random.RandomState(seed=config.seed)
    configure_logging(config.logging_conf_file)

    # INFO: получаем датасет с вопросами и контекстами
    # и логпробами по конкретной модели
    datasets_info = Dataset.list_datasets(
        dataset_project="RAG_Metrics",
        partial_name="Build logprobs",  # Ищет точное или частичное совпадение
    )
    required_tags = {config.llm.model, str(config.llm.count_logprobs)}
    matched_datasets = [
        d for d in datasets_info if required_tags.issubset(set(d.get("tags", [])))
    ]
    if matched_datasets:
        target_dataset_id = matched_datasets[0]["id"]
        dataset = Dataset.get(dataset_id=target_dataset_id, alias="logprobs_dataset")
        logger.info(f"Успешно загружен датасет с тегами: {dataset.tags}")
    else:
        logger.error("Датасет с такими тегами не найден.")
        raise RuntimeError

    dataset_path = dataset.get_local_copy()
    _qa_set_file = os.path.join(dataset_path, "dataset_QA.csv")
    passages_file = os.path.join(dataset_path, "passages.json")
    logprobs_file = os.path.join(dataset_path, "logprobs.parquet")

    table: Table = pq.read_table(logprobs_file)
    with open(passages_file, "r") as f:
        passages = json.load(f)

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

        context: str = passages.get(passage_id, "")

        try:
            logprob_parts: LogprobParts = map_logprobs2parts(
                prompt_logprob=prompt_logprob,
                top_logprob=top_logprob,
                context=context,
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
    main(parser.parse_args())
