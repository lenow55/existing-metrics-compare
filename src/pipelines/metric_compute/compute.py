import argparse
import logging
import os

import pyarrow.parquet as pq
from clearml import Dataset, Task, TaskTypes
from pyarrow import Table

from src.evaluators.cmp_cme.config import AppSettings
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
    if not isinstance(args.type, str):
        raise RuntimeError("Bad argument for type value")
    if not isinstance(args.metric, str):
        raise RuntimeError("Bad argument for mertic Func")

    if args.metric not in METRICS_HUB:
        raise RuntimeError(f"Metric {args.metric} not found in HUB {METRICS_HUB}")

    tags: list[str] = []
    tags.append(args.type)
    tags.append(args.metric)
    if args.cross_model:
        tags.append("CrossModel")

    c_task: Task = Task.init(
        project_name="RAG_Metrics",
        task_name="Logprobs_Scoring",
        task_type=TaskTypes.data_processing,
        tags=tags,
        reuse_last_task_id=False,
    )
    c_task.set_comment("Вычисление скоров значимости по логпробам")
    with open(args.config, "r") as f:
        conf = AppSettings.model_validate_json(f.read())
    config_dict = conf.model_dump(mode="python")
    config_dict = c_task.connect(config_dict, name="Hyperparameters")
    config = AppSettings(**config_dict)  # pyright: ignore[reportAny]

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
    _passages_file = os.path.join(dataset_path, "passages.json")
    logprobs_file = os.path.join(dataset_path, "logprobs_parted.parquet")

    table: Table = pq.read_table(logprobs_file)

    eval_ids = table.column("eval_id")
    instruct_p = table.column("instruct")
    context_p = table.column("context")
    question_p = table.column("question")
    answer_p = table.column("answer")

    results: list[ExperimentResult] = []
    target_func = METRICS_HUB[args.metric]
    logger.info(f"User {args.metric} function for scoring")
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
        dataset_tags=[
            "parquet",
            config.llm.model,
            str(config.llm.count_logprobs),
            args.metric,
        ],
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
    _ = parser.add_argument(
        "-m",
        "--metric",
        type=str,
        required=True,
        help="Имя функции метрики",
    )

    main(parser.parse_args())
