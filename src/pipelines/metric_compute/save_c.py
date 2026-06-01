import logging
import os
from dataclasses import dataclass
from tempfile import mkdtemp
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.schemas import (
    TA_list_metrics,
    TextUnitMetric,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentResult:
    eval_id: int
    instruct: list[TextUnitMetric]
    context: list[TextUnitMetric]
    question: list[TextUnitMetric]
    answer: list[TextUnitMetric]


def store_parquet(results: list[ExperimentResult]) -> tuple[str, pa.Table]:
    rows: list[dict[str, Any]] = []

    for r in results:
        # Сериализуем данные напрямую из полей объекта, используя правильные адаптеры.
        # dump_json() сразу валидирует и превращает в bytes, декодируем в str для Parquet.
        row = {
            "eval_id": r.eval_id,
            "instruct": TA_list_metrics.dump_json(r.instruct).decode("utf-8"),
            "context": TA_list_metrics.dump_json(r.context).decode("utf-8"),
            "question": TA_list_metrics.dump_json(r.question).decode("utf-8"),
            "answer": TA_list_metrics.dump_json(r.answer).decode("utf-8"),
        }
        rows.append(row)

    # Описываем схему данных PyArrow
    schema = pa.schema(
        [
            ("eval_id", pa.int64()),
            ("instruct", pa.large_string()),  # JSON-строка
            ("context", pa.large_string()),  # JSON-строка
            ("question", pa.large_string()),  # JSON-строка
            ("answer", pa.large_string()),  # JSON-строка
        ]
    )

    # Создаем таблицу напрямую из списка словарей БЕЗ pandas
    table = pa.Table.from_pylist(rows, schema=schema)

    # Размер датасета в памяти до экспорта в parquet
    size_mb = table.nbytes / (1024 * 1024)
    logger.info("Размер датасета до экспорта в parquet: %.2f MB", size_mb)

    # Создаем уникальную временную директорию
    temp_dir = mkdtemp(suffix="_p_logprobs")

    # Формируем полный путь к файлу
    file_path = os.path.join(temp_dir, "scores.parquet")

    # Записываем таблицу по указанному пути
    pq.write_table(
        table,
        file_path,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
    )

    return temp_dir, table
