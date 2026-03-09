from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, auc


def plot_precision_recall(
    binary_true: pd.Series,
    model_probas: Dict[str, np.array],
    figsize: Tuple[int, int] = (10, 10),
) -> plt.Figure:
    """
    :param binary_true: - actual values of the target variable. Indicates if the asset was manipulated or not
    :param model_probas: - predicted probabilities of the target variable. Keys are models' names that produced probabilities
    :param figsize: - size of the figure in inches
    :return: - plt.Figure with Precision Recall curve showing the performance of the model for different thresholds
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    for model_name, y_proba in model_probas.items():
        precision, recall, _ = precision_recall_curve(
            y_true=binary_true, y_score=y_proba
        )
        PrecisionRecallDisplay(precision=precision, recall=recall).plot(
            ax=ax,
            label=f"PRAUC: {model_name} - {auc(x=recall, y=precision):.4f}",  # type: ignore
        )

    # add isoquants for f1-score
    f_scores = np.linspace(0.1, 0.8, num=10)

    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = ax.plot(x[y >= 0], y[y >= 0], color="blue", alpha=0.2)
        ax.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))  # type: ignore

    plt.legend(loc="upper right")
    plt.title("Precision Recall curves")

    return fig
