import logging
import numpy as np
import numpy.typing as npt
import pandas as pd
from clearml import Logger as CLogger
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from src.utils.visual import plot_pr_binary, plot_roc_auc_binary, plot_true_lie_distrib

logger = logging.getLogger(__name__)


def log_bin_report(
    logger_c: CLogger,
    test_report_d: dict[str, dict[str, float]],
    target_names: list[str],
    skip_support: bool = False,
):
    # INFO: 2. логируем метрики pr, rc, f1 по каждому классу отдельно
    # Проходимся по именам классов (для бинарной это обычно 2 класса, например ["Class_0", "Class_1"])
    for class_name in target_names:
        for metric_n, metric_v in test_report_d[class_name].items():
            if metric_n == "support" and skip_support:
                # Пропускаем support, так как это просто количество сэмплов, а не метрика
                continue

            # Формируем имя с учетом названия класса, например: test_Class_0_precision
            # Заменяем пробелы на подчеркивания на случай, если в target_names есть пробелы
            safe_class_name = class_name.replace(" ", "_")

            logger_c.report_single_value(
                name=f"test_{safe_class_name}_{metric_n}", value=round(metric_v, 3)
            )


def classifier_report_plan(
    y_true: npt.NDArray[np.int64],
    y_score: npt.NDArray[np.float32],
    index: npt.NDArray[np.str_],
    logger_c: CLogger,
    metric_name: str,
    show_hist: bool = False,
    bin_size: float = 0.01,
):
    # INFO: 0. вычисляем лучший трэшхолд
    # 1. Считаем FPR, TPR и пороги
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # 2. Вычисляем Youden's J statistic для каждого порога
    j_scores = tpr - fpr

    # 3. Находим индекс максимального значения
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    logger.info(f"Лучший порог по ROC (Youden's J): {best_threshold:.4f}")
    logger_c.report_single_value(
        name="best_roc_threshold", value=round(best_threshold, 4)
    )

    y_pred = (y_score >= best_threshold).astype(int)

    # INFO: 1. строим классификационный отчёт
    target_names = ["Ложь", "Истина"]
    test_report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
    )
    if isinstance(test_report, str):
        logger.info("\n" + test_report)

    test_report_d = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
    )
    if not isinstance(test_report_d, dict):
        raise

    report_d = pd.DataFrame.from_dict(test_report_d)

    logger_c.report_table(
        title="Eval Report",
        series="classification report",
        table_plot=report_d.T.round(2),
    )

    # INFO: 2. Логируем метрики по классам
    log_bin_report(
        logger_c=logger_c, test_report_d=test_report_d, target_names=target_names
    )

    # INFO: 3. Логируем PR-ROC кривые
    fig_pr, ap = plot_pr_binary(
        y_true=y_true,
        y_score=y_score,
        class_name="Истина",
    )
    logger_c.report_plotly(
        title="Eval Report",
        series="PR кривая",
        figure=fig_pr,
    )
    logger_c.report_single_value("AP", ap)
    fig_roc, auc = plot_roc_auc_binary(
        y_true=y_true,
        y_score=y_score,
        class_name="Истина",
    )
    logger_c.report_single_value("AUC", auc)
    logger_c.report_plotly(
        title="Eval Report",
        series="ROC кривая",
        figure=fig_roc,
    )

    # INFO: 4. Логируем confusionMatrix
    # 2. Переводим вероятности в бинарные предсказания (0 или 1) по порогу 0.5

    # 3. Вычисляем саму матрицу ошибок
    cm = confusion_matrix(y_true, y_pred)

    # 4. Логируем готовую матрицу в ClearML
    logger_c.report_confusion_matrix(
        title="Eval Report",
        series="Матрица Коллизий",  # Название графика
        matrix=cm,  # Передаем посчитанную матрицу
        xaxis="Прогноз",
        yaxis="Реальность",
        xlabels=target_names,
        ylabels=target_names,
        extra_layout={
            "texttemplate": "%{z}",
            "colorscale": [
                [0.00, "white"],
                [0.40, "white"],  # до ~65% диапазона чисел — почти белый
                [0.65, "rgb(220,235,250)"],
                [0.75, "rgb(140,185,225)"],
                [1.00, "rgb(40,100,170)"],
            ],
            "textfont": {"size": 24},
            "font": {"size": 16},
        },
    )

    # INFO: 5. Распределение Реально правильных и неправильных
    # с трэшхолдом
    # Create distplot with curve_type set to 'normal'
    fig_dist = plot_true_lie_distrib(
        y_true=y_true,
        y_score=y_score,
        eval_ids=index,
        target_names=["Истина", "Ложь"],
        show_hist=show_hist,
        bin_size=bin_size,
    )
    logger_c.report_plotly(
        title=f"{metric_name} Report",
        series="Распределение близости ответов",
        figure=fig_dist,
    )
