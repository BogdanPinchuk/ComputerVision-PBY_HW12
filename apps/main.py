import cv2
import numpy as np
import apps.reporter as rpt
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import display
from IPython.core.display import Markdown
from pandas.io.formats.style import Styler
from packaging.requirements import Requirement
from importlib.metadata import PackageNotFoundError, version
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, roc_auc_score, average_precision_score,
                             precision_recall_curve, roc_curve)


def load_image(file_name: str) -> np.ndarray:
    """
    Load image by file path
    :param file_name: file name
    :return: image
    """
    file_path = Path(file_name)

    # Load an image (you can freely choose any image you like)
    if file_path.exists():
        img_bgr = cv2.imread(str(file_path))
    else:
        raise FileNotFoundError(f"File '{file_name}' doesn't exist!")

    if img_bgr is None:
        raise ValueError("Incorrect input data! Could not decode image: '{file_name}'")

    # Convert it to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    return img_rgb


def is_package_satisfied(req_str: str) -> bool:
    try:
        req = Requirement(req_str)
        installed_ver = version(req.name)
        if req.specifier:
            return installed_ver in req.specifier
        return True
    except PackageNotFoundError:
        return False


def calc_class_metrics(y_test, y_pred, y_prob=None, charts=True, figsize: tuple[float, float] = (10, 4)) -> Styler:
    """
    Calc and print a classifier metrics
    :param y_test: original target test set
    :param y_pred: predicted set
    :param y_prob: probabilities set
    :param charts: show charts
    :param figsize: figure size
    :return: DataFrame styler
    """
    rp = rpt.Reporter()
    rp.tolerance = 4

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    rp.add_item("Confusion Matrix", rp.format_matrix(cm))
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    rp.add_item("Accuracy\n(Точність)", rp.format_value(accuracy))
    # Precision
    precision = precision_score(y_test, y_pred)
    rp.add_item("Precision\n(Влучність)", rp.format_value(precision))
    # Recall
    recall = recall_score(y_test, y_pred)
    rp.add_item("Recall\n(Повнота)", rp.format_value(recall))
    # F1-score
    f1 = f1_score(y_test, y_pred)
    rp.add_item("F1-score", rp.format_value(f1))

    if y_prob is not None:
        # ROC-AUC
        auc = roc_auc_score(y_test, y_prob)
        rp.add_item("Receiver Operating Characteristic\n(ROC-AUC)", rp.format_value(auc))
        # AP
        ap = average_precision_score(y_test, y_prob)
        rp.add_item("Average Precision (AP)", rp.format_value(ap))

    df = rp.get_pd_report()

    # Print results
    rp.print_pd_report(f"Metrics")
    print(classification_report(y_test, y_pred))

    # Graphic results
    if y_prob is not None and charts:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)

        _, axes = plt.subplots(1, 2, figsize=figsize)

        ax = axes[0]
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], '--')
        ax.fill_between(fpr, tpr, alpha=0.25)
        ax.grid(axis='both', visible=True, which='major', ls='--', linewidth=1.0, color='tab:gray')
        # ax.minorticks_on()
        # ax.grid(axis='both', visible=True, which='minor', ls=':', linewidth=0.5, color='tab:green')
        ax.set_title("ROC Curve", pad=10, loc='center', color='black')
        ax.set_xlabel("FPR", labelpad=10, loc='center', color='black')
        ax.set_ylabel("RPT", labelpad=10, loc='center', color='black')

        ax = axes[1]
        ax.plot(rec, prec)
        ax.plot([1, 0], [0, 1], '--')
        ax.fill_between(rec, prec, alpha=0.25)
        ax.grid(axis='both', visible=True, which='major', ls='--', linewidth=1.0, color='tab:gray')
        # ax.minorticks_on()
        # ax.grid(axis='both', visible=True, which='minor', ls=':', linewidth=0.5, color='tab:green')
        ax.set_title("Precision-Recall Curve", pad=10, loc='center', color='black')
        ax.set_xlabel("Recall", labelpad=10, loc='center', color='black')
        ax.set_ylabel("Precision", labelpad=10, loc='center', color='black')

        plt.suptitle(f"Model Performance Curves")

        plt.tight_layout()
        plt.show()

    return df
