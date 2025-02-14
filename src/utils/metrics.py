from collections import defaultdict
import torch


def iou_score(
    y_true: torch.Tensor, y_pred: torch.Tensor, logits: bool = False
) -> torch.Tensor:
    """
    Returns intersection over union score per class.
    """
    if logits:
        y_pred = (y_pred > 0).float()
    else:
        y_pred = (y_pred > 0.5).float()
    intersection = y_pred * y_true
    union = y_pred + y_true - intersection
    iou = intersection.sum(dim=(-1, -2)) / union.sum(dim=(-1, -2))
    return iou


class MetricCalc:
    def __init__(self, metric_dict: dict[str, callable], cache_metrics: bool = True):
        self.metric_dict = metric_dict
        self.cache_metrics = cache_metrics
        self.cache = {m: torch.tensor(0.0) for m in self.metric_dict}
        self.num_examples = 0

    def add_metric(self, name: str, metric: callable):
        self.metric_dict[name] = metric

    def calc_metrics(
        self, y_true: torch.Tensor, y_pred: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        resulting_metrics = dict()
        for metric_name, metric_func in self.metric_dict.items():
            resulting_metrics[metric_name] = metric_func(y_true, y_pred)

        return resulting_metrics
