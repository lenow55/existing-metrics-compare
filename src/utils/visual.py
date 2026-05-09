import numpy as np
import numpy.typing as npt
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)


def plot_pr_binary(
    y_true: npt.NDArray[np.int64],
    y_score: npt.NDArray[np.float32],
    class_name: str = "Positive Class",
) -> tuple[go.Figure, float]:
    fig_pr = go.Figure()

    # 1. Вычисляем метрики для единственного класса
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    # 2. Добавляем кривую на график
    fig_pr = fig_pr.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            name=f"{class_name} (AP={ap:.3f})",
            mode="lines",
            line=dict(color="navy", width=3),
        )
    )

    # 3. Настраиваем внешний вид
    fig_pr = fig_pr.update_layout(
        title="PR кривая",
        xaxis_title="Полнота",
        yaxis_title="Точность",
        xaxis=dict(range=[0.0, 1.0]),
        yaxis=dict(range=[0.0, 1.05]),
        template="plotly_white",  # Базовый белый шаблон
        plot_bgcolor="white",  # Внутренний фон (где рисуются линии)
        paper_bgcolor="white",  # Внешний фон (где заголовки и легенда)
    )

    return fig_pr, float(ap)


def plot_roc_auc_binary(
    y_true: npt.NDArray[np.int64],
    y_score: npt.NDArray[np.float32],
    class_name: str = "Positive Class",
) -> tuple[go.Figure, float]:
    fig_roc = go.Figure()

    # 1. Считаем метрики для единственного класса
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # 2. Строим ROC-кривую
    fig_roc = fig_roc.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            name=f"{class_name} (AUC={roc_auc:.3f})",
            mode="lines",
            line=dict(color="deeppink", width=3),
        )
    )

    # 3. Добавляем диагональную линию (случайное угадывание)
    fig_roc = fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="black", dash="dash"),
            showlegend=False,
        )
    )

    # 4. Настраиваем внешний вид с максимально белым фоном
    fig_roc = fig_roc.update_layout(
        title="ROC-кривая",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        # Настройки осей для белого фона
        xaxis=dict(range=[0.0, 1.0]),
        yaxis=dict(range=[0.0, 1.05]),
        # Шаблон и цвета фона
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig_roc, float(roc_auc)


def plot_true_lie_distrib(
    y_true: npt.NDArray[np.int64],
    y_score: npt.NDArray[np.float32],
    eval_ids: npt.NDArray[np.str_],
    target_names: list[str],
    show_hist: bool = False,
):
    true_idx = np.flatnonzero(y_true)
    bad_idx = np.argwhere(y_true == 0).flatten()
    fig = ff.create_distplot(
        [y_score[true_idx], y_score[bad_idx]],
        target_names,
        curve_type="kde",
        # curve_type="normal",  # override default 'kde'
        bin_size=0.01,
        rug_text=[eval_ids[true_idx], eval_ids[bad_idx]],
        show_hist=show_hist,
    )

    # Add title
    fig = fig.update_layout(title_text="Распределение близости ответов")
    # 3. Настраиваем ширину, высоту и другие параметры макета
    fig = fig.update_layout(
        margin=dict(l=50, r=50, t=50, b=50),  # Отступы от краев (опционально)
    )
    return fig
