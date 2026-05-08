from clearml import Logger


def log_bin_report(
    logger_c: Logger,
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
