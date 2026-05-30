import asyncio
import json
import logging
import os
import traceback
from typing import cast

import numpy as np
import pandas as pd
from clearml import Dataset, Task, TaskTypes
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from src.config import ChatLLMConfig
from src.evaluators.schemas import EvalIn
from src.utils.base import (
    calculate_prompt_logprobs,
    configure_logging,
    create_openai_client,
)
from src.utils.startup import init_config

from .config import AppSettings
from .save import store_parquet
from .schemas import ExperimentResult

logger = logging.getLogger(__name__)


async def logprob_generation(
    *,
    eval: EvalIn,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    config: ChatLLMConfig,
    passages: dict[str, str],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
):
    messages_prefix: list[dict[str, str]] = [
        {
            "role": "user",
            "content": f"context: {passages[eval['passage_id']]}\nquestion: {eval['question']}",
        }
    ]
    prefix_text = tokenizer.apply_chat_template(
        messages_prefix,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(prefix_text, str):
        raise TypeError(
            f"Expected str from apply_chat_template, got {type(prefix_text).__name__}"
        )

    # Кодируем получившуюся строку для получения количества токенов
    prefix_tokens = tokenizer.encode(prefix_text)
    prefix_length = len(prefix_tokens)

    messages: list[ChatCompletionMessageParam] = [
        cast(ChatCompletionMessageParam, cast(object, i)) for i in messages_prefix
    ]
    messages.append({"role": "assistant", "content": eval["answer"]})
    async with semaphore:
        try:
            _, logprobs = await calculate_prompt_logprobs(
                messages=messages, client=client, config=config
            )
            return ExperimentResult(
                eval_id=eval["eval_id"],
                answer=eval["answer"],
                passage_id=eval["passage_id"],
                question=eval["question"],
                prompt_logprob=logprobs,
                prefix_length=prefix_length,
                label=eval["label"],
                ok=True,
            )
        except Exception as exc:  # noqa: BLE001 — намеренно широкий перехват для статистики
            logger.error("Calculate prompt logprob failed")
            return ExperimentResult(
                eval_id=eval["eval_id"],
                answer=eval["answer"],
                passage_id=eval["passage_id"],
                question=eval["question"],
                label=eval["label"],
                prompt_logprob=[],
                prefix_length=prefix_length,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )


async def main():
    c_task: Task = Task.init(
        project_name="RAG_Metrics",
        task_name="Build logprobs",
        task_type=TaskTypes.testing,
        tags=["build", "Logprobs", "CrossModel"],
        reuse_last_task_id=False,
    )
    c_task.set_comment("Вычисление датасета логпробов на модели CrossModel")

    config = init_config(conf_type=AppSettings, task=c_task)
    random_state = np.random.RandomState(seed=config.seed)
    configure_logging(config.logging_conf_file)

    client: AsyncOpenAI = create_openai_client(config=config.llm)
    sem = asyncio.Semaphore(config.llm.async_cals)

    # получаем датасет
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

    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = (
        AutoTokenizer.from_pretrained(config.llm.model)
    )

    tasks: list[asyncio.Task[ExperimentResult]] = []
    for idx, row in qa_dataset.iterrows():
        if not isinstance(idx, int):
            raise RuntimeError("Bad index type")
        tasks.append(
            asyncio.create_task(
                logprob_generation(
                    eval={
                        "eval_id": idx,
                        "answer": str(row["answer"]),
                        "passage_id": str(int(row["passage_id"])),
                        "question": str(row["question"]),
                        "reference": str(row["reference"]),
                        "label": int(row["label"]),
                    },
                    client=client,
                    semaphore=sem,
                    config=config.llm,
                    passages=passages,
                    tokenizer=tokenizer,
                )
            )
        )

    results: list[ExperimentResult] = []
    counter = 0
    try:
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                results.append(result)
            except Exception as e:
                logger.warning(f"Error when logprob calculation, record will skip: {e}")
                logger.debug(traceback.format_exc())
                continue
            finally:
                counter = counter + 1
                if counter % 100 == 0:
                    logger.info(f"Processed {counter}/{len(tasks)}")
    except asyncio.exceptions.CancelledError:
        logger.error("Получен Ctrl+C, отменяем задачи...")
        for t in tasks:
            _ = t.cancel()
        _ = await asyncio.gather(*tasks, return_exceptions=True)
        logger.error("Все задачи корректно завершены")
        _ = c_task.flush()
        _ = c_task.mark_completed(status_message="stopped")
        c_task.close()
        exit(1)

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
