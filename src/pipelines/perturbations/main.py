import argparse
import json
import logging
import os
from dataclasses import dataclass
from tempfile import mkdtemp
from typing import TypedDict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from clearml import Dataset, Task, TaskTypes
from pyarrow import Table
from transformers import pipeline

from src.evaluators.cmp_cme.config import AppSettings
from src.metrics import METRICS_HUB
from src.schemas import TA_list_metrics
from src.utils.base import (
    configure_logging,
)

from .utils import replace_masks_with_inverse_probability

classifier = pipeline("fill-mask", model="xlm-roberta-base")


logger = logging.getLogger(__name__)


class PtbResult(TypedDict):
    id: int
    context: str
    change_set: dict[int, str]


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
    "-s",
    "--cross-model",
    action="store_true",
    required=False,
    default=False,
    help="Флаг генерации CrossModel",
)
_ = parser.add_argument(
    "-r",
    "--reverse",
    action="store_false",
    required=False,
    default=True,
    help="Как сортировать метрику. По умолчанию по убыванию",
)
_ = parser.add_argument(
    "-p",
    "--perturbs",
    type=int,
    required=True,
    help="Количество слов для замены",
)
_ = parser.add_argument(
    "-m",
    "--metric",
    type=str,
    required=True,
    help="Имя функции метрики",
)


def main(args: argparse.Namespace):
    if not isinstance(args.config, str):
        raise RuntimeError("Bad argument for config path value")
    if not isinstance(args.cross_model, bool):
        raise RuntimeError("Bad argument for cross-model value")
    if not isinstance(args.metric, str):
        raise RuntimeError("Bad argument for mertic Func")
    if not isinstance(args.perturbs, int):
        raise RuntimeError("Bad argument for perturbs")
    if not isinstance(args.reverse, bool):
        raise RuntimeError("Bad argument for reverse")

    if args.metric not in METRICS_HUB:
        raise RuntimeError(f"Metric {args.metric} not found in HUB {METRICS_HUB}")

    type_task: str = "PTB"
    tags: list[str] = []
    tags.append(type_task)
    tags.append(args.metric)
    # TODO: потом будет задаваться динамически
    tags.append("xlm-roberta-base")
    if args.cross_model:
        tags.append("CrossModel")

    c_task: Task = Task.init(
        project_name="RAG_Metrics",
        task_name="ApplyPerturb",
        task_type=TaskTypes.data_processing,
        tags=tags,
        reuse_last_task_id=False,
    )
    c_task.set_comment("Замена значимых слов")
    with open(args.config, "r") as f:
        conf = AppSettings.model_validate_json(f.read())
    config_dict = conf.model_dump(mode="python")
    config_dict = c_task.connect(config_dict, name="Hyperparameters")
    config = AppSettings(**config_dict)  # pyright: ignore[reportAny]

    configure_logging(config.logging_conf_file)
    rng = np.random.default_rng(seed=config.seed)

    # INFO: получаем датасет с вопросами и контекстами
    # и логпробами по конкретной модели
    datasets_info = Dataset.list_datasets(
        dataset_project="RAG_Metrics",
        partial_name="Logprobs_Mapping",  # Ищет точное или частичное совпадение
    )

    required_tags = {
        config.llm.model,
        str(config.llm.count_logprobs),
        args.metric,
        "CTX",
    }
    if args.cross_model:
        required_tags.add("CrossModel")
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
    qa_set_file = os.path.join(dataset_path, "dataset_QA.csv")
    scores_file = os.path.join(dataset_path, "scores.parquet")

    qa_dataset = pd.read_csv(qa_set_file, index_col=0)
    qa_dataset.index = qa_dataset.index.astype(int)
    logger.info(f"QA readed from {qa_set_file} shape: {qa_dataset.shape}")

    table: Table = pq.read_table(scores_file)

    eval_ids = table.column("eval_id")
    context_p = table.column("context")

    results: list[PtbResult] = []
    counter = 0
    for (
        eval_id,
        context,
    ) in zip(
        eval_ids,
        context_p,
    ):
        eval_id: int = int(eval_id)
        context = TA_list_metrics.validate_json(context.as_py())

        top_text_units = sorted(context, key=lambda x: x["value"], reverse=args.reverse)
        # Получили отсортированные части текста
        top_text_idx: list[int] = [u["index"] for u in top_text_units]
        top_text_idx = top_text_idx[: args.perturbs]
        top_text_idx_s = set(top_text_idx)

        restored_ctx: list[str] = []
        for u in context:
            if u["index"] in top_text_idx_s:
                restored_ctx.append("<mask>")
            else:
                restored_ctx.append(u["text_unit"])
        masked_ctx: str = "".join(restored_ctx)

        try:
            perturbed, changes = replace_masks_with_inverse_probability(
                text=masked_ctx,
                classifier=classifier,
                rng=rng,
            )

        except RuntimeError:
            logger.warning(f"Bad scoring in eval_id: {eval_id}")
            continue
        finally:
            counter = counter + 1
            if counter % 100 == 0:
                logger.info(f"Processed {counter}/{table.num_rows}")

        change_set: dict[int, str] = {}
        for idx, change in zip(sorted(top_text_idx, reverse=True), changes):
            change_set.update({idx: change})

        results.append(PtbResult(id=eval_id, context=perturbed, change_set=change_set))

    df = pd.DataFrame(
        {
            "id": r["id"],
            "context": r["context"],
            "change_set": json.dumps(r["change_set"], ensure_ascii=False),
        }
        for r in results
    ).set_index("id")

    merged_QA_set = qa_dataset.join(df, on="id").set_index("id")
    temp_dir = mkdtemp(suffix="_p_logprobs")
    merged_QA_path = os.path.join(temp_dir, "dataset_QA.csv")
    merged_QA_set.to_csv(merged_QA_path, index_label="id")

    new_dataset = Dataset.create(
        dataset_project="RAG_Metrics",
        dataset_name="ApplyPerturb",
        dataset_tags=[
            config.llm.model,
            str(config.llm.count_logprobs),
            str(args.perturbs),
        ],
        use_current_task=True,
        description="Датасет с метриками по логпробам",
        parent_datasets=[dataset],
    )
    _ = new_dataset.add_files(path=temp_dir)
    _ = new_dataset.upload()
    _ = new_dataset.finalize()

    _ = c_task.flush(wait_for_uploads=True)
    _ = c_task.mark_completed(status_message="completed")
    c_task.close()


if __name__ == "__main__":
    main(parser.parse_args())
