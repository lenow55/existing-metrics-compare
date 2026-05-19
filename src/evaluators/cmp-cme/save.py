import os
from dataclasses import asdict
from tempfile import mkdtemp
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.schemas import TA_logprob_list

from .schemas import ExperimentResult


def store_parquet(results: list[ExperimentResult]) -> tuple[str, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for r in results:
        d = asdict(r)
        # Валидация структуры через pydantic TypeAdapter и сериализация в JSON.
        # dump_json возвращает bytes -> декодируем в str для хранения в колонке.
        d["prompt_logprob"] = TA_logprob_list.dump_json(
            TA_logprob_list.validate_python(d["prompt_logprob"])
        ).decode("utf-8")
        rows.append(d)

    df = pd.DataFrame(rows)

    schema = pa.schema(
        [
            ("eval_id", pa.int64()),
            ("passage_id", pa.string()),
            ("label", pa.int32()),
            ("question", pa.string()),
            ("answer", pa.string()),
            ("prompt_logprob", pa.large_string()),  # JSON
            ("prefix_length", pa.int32()),
            ("ok", pa.bool_()),
            ("error", pa.string()),
        ]
    )

    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)

    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)

    # Создаем уникальную временную директорию
    temp_dir = mkdtemp(suffix="_logprobs")

    # Формируем полный путь к файлу с нужным именем
    file_path = os.path.join(temp_dir, "logprobs.parquet")

    # Записываем таблицу по указанному пути
    pq.write_table(
        table,
        file_path,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
    )

    return temp_dir, df
