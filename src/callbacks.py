from sklearn import metrics
from lightning.pytorch import callbacks
import torch


def get_metrics_and_curves(metric_type, y_pred, y_true, threshold=0.5):
    """
    Calculate metrics and curves for a given metric type
    ROC: Receiver Operating Characteristic curve, metric = Area under the curve
    PR: Precision-Recall curve, metric = Area under the curve (Average precision)
    CM: Confusion Matrix, metric = F1 score

    Parameters
    ----------
    metric_type : str
        One of "ROC", "PR"
    y_pred : torch.Tensor
        Predicted labels
    y_true : torch.Tensor
        True labels

    Returns
    -------
    metric_value : float
        Value of the metric
    metric_disp : matplotlib.figure.Figure
        Figure of the curve/matrix
    """
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    if metric_type == "ROC":
        # Receiver Operating Characteristic Curve
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        roc_disp = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
        return roc_auc, roc_disp.figure_
    elif metric_type == "PR":
        # Precision-Recall Curve
        precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred, pos_label=1)
        pr_auc = metrics.auc(recall, precision)
        pr_disp = metrics.PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=pr_auc).plot()
        return pr_auc, pr_disp.figure_


class LogMetrics(callbacks.Callback):
    """
    Log metrics and curves for validation and training

    Scalars: ROC/val_AUC, ROC/train_AUC, PR/val_AUC, PR/train_AUC, CM/val_F1, CM/train_F1
    Images: ROC/val, ROC/train, PR/val, PR/train, CM/val, CM/train
    """
    def on_validation_epoch_end(self, trainer, pl_module):
        outputs = torch.cat([x[0] for x in pl_module.validation_step_outputs], dim=0)
        labels = torch.cat([x[1] for x in pl_module.validation_step_outputs], dim=0)
        for metric in ["ROC", "PR"]:
            metric_auc, metric_disp = get_metrics_and_curves(metric, outputs, labels)
            pl_module.log(f"{metric}/val_AUC", metric_auc)
            trainer.logger.experiment.add_figure(f"{metric}/val", metric_disp, global_step=trainer.global_step)

    def on_train_epoch_end(self, trainer, pl_module):
        outputs = torch.cat([x[0] for x in pl_module.train_step_outputs], dim=0)
        labels = torch.cat([x[1] for x in pl_module.train_step_outputs], dim=0)
        for metric in ["ROC", "PR"]:
            metric_auc, metric_disp = get_metrics_and_curves(metric, outputs, labels)
            pl_module.log(f"{metric}/train_AUC", metric_auc)
            trainer.logger.experiment.add_figure(f"{metric}/train", metric_disp, global_step=trainer.global_step)