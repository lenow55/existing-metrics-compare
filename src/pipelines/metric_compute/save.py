import logging
import os
from dataclasses import asdict, dataclass
from tempfile import mkdtemp
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.schemas import (
    LogprobParts,
    TA_logprob_list,
    TA_logprob_steps_list,
    TA_prompt_loprobs_list,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentResult(LogprobParts):
    eval_id: int


def store_parquet(results: list[ExperimentResult]) -> tuple[str, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for r in results:
        d = asdict(r)
        # Валидация структуры через pydantic TypeAdapter и сериализация в JSON.
        # dump_json возвращает bytes -> декодируем в str для хранения в колонке.
        d["instruct"] = TA_prompt_loprobs_list.dump_json(
            TA_prompt_loprobs_list.validate_python(d["instruct"])
        ).decode("utf-8")
        d["context"] = TA_prompt_loprobs_list.dump_json(
            TA_prompt_loprobs_list.validate_python(d["context"])
        ).decode("utf-8")
        d["question"] = TA_prompt_loprobs_list.dump_json(
            TA_prompt_loprobs_list.validate_python(d["question"])
        ).decode("utf-8")
        d["answer"] = TA_prompt_loprobs_list.dump_json(
            TA_prompt_loprobs_list.validate_python(d["answer"])
        ).decode("utf-8")
        rows.append(d)

    df = pd.DataFrame(rows)

    schema = pa.schema(
        [
            ("eval_id", pa.int64()),
            ("instruct", pa.large_string()),  # JSON
            ("context", pa.large_string()),  # JSON
            ("question", pa.large_string()),  # JSON
            ("answer", pa.large_string()),  # JSON
        ]
    )

    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)

    # Размер датасета в памяти до экспорта в parquet
    size_mb = table.nbytes / (1024 * 1024)
    logger.info("Размер датасета до экспорта в parquet: %.2f MB", size_mb)

    # Создаем уникальную временную директорию
    temp_dir = mkdtemp(suffix="_p_logprobs")

    # Формируем полный путь к файлу с нужным именем
    file_path = os.path.join(temp_dir, "logprobs_parted.parquet")

    # Записываем таблицу по указанному пути
    pq.write_table(
        table,
        file_path,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
    )

    return temp_dir, df
