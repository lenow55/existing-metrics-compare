import argparse
import logging
import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from clearml import Dataset, Logger, Task, TaskTypes
from sklearn.model_selection import train_test_split

from src.utils.base import configure_logging
from src.utils.report import classifier_report_plan

logger = logging.getLogger(__name__)


class ClearMLIterationLogger:
    """Callback CatBoost для логирования метрик каждой итерации в ClearML."""

    def __init__(self, clearml_logger: Logger):
        self._logger = clearml_logger

    def after_iteration(self, info) -> bool:
        # info.metrics: dict[str, dict[str, list[float]]]
        # Структура: {dataset_name: {metric_name: [значения по итерациям]}}
        iteration = info.iteration
        for dataset_name, metrics in info.metrics.items():
            for metric_name, values in metrics.items():
                if not values:
                    continue
                self._logger.report_scalar(
                    title=metric_name,
                    series=dataset_name,
                    value=float(values[-1]),
                    iteration=iteration,
                )
        return True


configure_logging("logging_config.json")

parser = argparse.ArgumentParser()
_ = parser.add_argument(
    "-d",
    "--dataset",
    type=str,
    required=True,
    help="ID датасета для обучения",
)


def main(args: argparse.Namespace):
    if not isinstance(args.dataset, str):
        raise RuntimeError("Не указан ID датасета для обучения")

    c_task: Task = Task.init(
        project_name="RAG_Metrics",
        task_name="CatBoost",
        task_type=TaskTypes.training,
        reuse_last_task_id=False,
    )
    c_task.set_comment("Обучение CatBoost (оптимизация кросс-энтропии)")

    dataset = Dataset.get(
        dataset_id=args.dataset,
        alias="train_dataset",
    )
    dataset_path = dataset.get_local_copy()
    c_task.set_tags(tags=dataset.tags)

    train_file = os.path.join(dataset_path, "train.parquet")
    train_df = pd.read_parquet(train_file, engine="pyarrow")

    X = train_df.drop(columns="Y").values
    y = train_df["Y"].values
    indices = np.arange(len(train_df))

    # Разделение данных (80%% обучение, 20%% тест)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        indices,
        test_size=0.2,
        random_state=42,
        stratify=y,  # pyright: ignore[reportArgumentType]
    )

    # Инициализация модели CatBoost.
    # loss_function="Logloss" — бинарная кросс-энтропия.
    # random_seed обеспечивает воспроизводимость.
    clf = CatBoostClassifier(
        iterations=1000,
        loss_function="Logloss",
        eval_metric="Logloss",
        custom_metric=["AUC", "f1"],
        random_seed=42,
        thread_count=-1,  # использование всех ядер процессора
        verbose=False,
        allow_writing_files=False,
    )

    # Подготовка Pool для train/eval
    train_pool = Pool(data=X_train, label=y_train)
    eval_pool = Pool(data=X_test, label=y_test)

    logger_c = c_task.get_logger()

    # Обучение с логированием каждой итерации в ClearML
    _ = clf.fit(
        train_pool,
        eval_set=eval_pool,
        callbacks=[ClearMLIterationLogger(logger_c)],
    )

    # Логирование параметров обученной модели в ClearML
    _ = c_task.connect(clf.get_params(), name="CatBoost")
    fitted_attrs = {
        "tree_count_": int(clf.tree_count_),
        "best_iteration_": int(clf.get_best_iteration() or 0),
        "best_score_": clf.get_best_score(),
        "feature_count": int(X_train.shape[1]),
    }
    _ = c_task.connect(fitted_attrs, name="CatBoost_fitted")

    # Оценка качества
    y_pred = clf.predict_proba(X_test)

    if not isinstance(y_test, np.ndarray):
        raise
    if not isinstance(y_pred, np.ndarray):
        raise

    # Сохранение предсказаний с индексами как артефакт
    pred_df = pd.DataFrame(
        y_pred,
        index=train_df.index.values[idx_test],
        columns=[f"proba_class_{i}" for i in range(y_pred.shape[1])],
    )
    pred_df["y_true"] = y_test
    pred_df.index.name = "eval_id"

    _ = c_task.upload_artifact(
        name="test_predictions",
        artifact_object=pred_df,
    )

    classifier_report_plan(
        y_true=y_test,
        y_score=y_pred[:, 1],
        index=train_df.index.values[idx_test],
        logger_c=logger_c,
        metric_name="Предлагаемая Модель",
        show_hist=True,
    )

    _ = c_task.flush(wait_for_uploads=True)
    _ = c_task.mark_completed(status_message="completed")
    c_task.close()


if __name__ == "__main__":
    main(parser.parse_args())
