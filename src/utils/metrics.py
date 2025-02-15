from torchmetrics import Metric
import torch


class IOUScore(Metric):
    def __init__(
        self,
        num_classes: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.add_state(
            "iou_score", default=torch.zeros(num_classes), dist_reduce_fx="sum"
        )
        self.add_state("num_examples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")
        intersection = torch.logical_and(preds, target).float().sum(dim=(-1, -2))
        union = torch.logical_or(preds, target).float().sum(dim=(-1, -2))
        self.iou_score += (intersection / union).sum(dim=0)
        self.num_examples += target.shape[0]

    def compute(self) -> torch.Tensor:
        return self.iou_score / self.num_examples


if __name__ == "__main__":

    iou = IOUScore(num_classes=3)
    x = torch.ones(4, 3, 100, 100)
    y = torch.ones(4, 3, 100, 100)
    for i in range(10):
        iou.update(x, y)
        print(iou.iou_score.shape)
    print(iou.compute())
