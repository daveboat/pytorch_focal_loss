import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    One possible pytorch implementation of focal loss (https://arxiv.org/abs/1708.02002), for multiclass classification.
    This module is intended to be easily swappable with nn.CrossEntropyLoss.

    If with_logits is true, then input is expected to be a tensor of raw logits, and a softmax is applied
    If with_logits is false, then input is expected to be a tensor of probabiltiies

    target is expected to be a batch of integer targets (i.e. sparse, not one-hot). This is the same behavior as
    nn.CrossEntropyLoss.

    This loss also ignores contributions where target == ignore_index, in the same way as nn.CrossEntropyLoss

    batch behaviour: reduction = 'none', 'mean', 'sum'
    """

    def __init__(self, gamma=1, eps=1e-7, with_logits=True, ignore_index=-100, reduction='mean'):
        super().__init__()

        assert reduction in ['none', 'mean', 'sum'], 'FocalLoss: reduction must be one of [\'none\', \'mean\', \'sum\']'

        self.gamma = gamma
        self.eps = eps
        self.with_logits = with_logits
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        return focal_loss(input, target, self.gamma, self.eps, self.with_logits, self.ignore_index, self.reduction)


def focal_loss(input, target, gamma=1, eps=1e-7, with_logits=True, ignore_index=-100, reduction='mean'):
    """
    A function version of focal loss, meant to be easily swappable with F.cross_entropy. The equation implemented here
    is L_{focal} = - \sum (1 - p_{target})^\gamma p_{target} \log p_{pred}

    If with_logits is true, then input is expected to be a tensor of raw logits, and a softmax is applied
    If with_logits is false, then input is expected to be a tensor of probabiltiies (softmax previously applied)

    target is expected to be a batch of integer targets (i.e. sparse, not one-hot). This is the same behavior as
    nn.CrossEntropyLoss.

    Loss is ignored at indices where the target is equal to ignore_index

    batch behaviour: reduction = 'none', 'mean', 'sum'
    """
    y = F.one_hot(target, input.size(-1))
    if with_logits:
        pt = F.softmax(input, dim=-1)
    else:
        pt = input
        pt = pt.clamp(eps, 1. - eps)  # a hack-y way to prevent taking the log of a zero, because we are dealing with probabilities directly.

    loss = -y * torch.log(pt)  # cross entropy
    loss *= (1 - pt) ** gamma  # focal loss factor
    loss = torch.sum(loss, dim=-1)

    # mask the logits so that values at indices which are equal to ignore_index are ignored
    loss = loss[target != ignore_index]

    # batch reduction
    if reduction == 'mean':
        return torch.mean(loss, dim=-1)
    elif reduction == 'sum':
        return torch.sum(loss, dim=-1)
    else:  # 'none'
        return loss

if __name__ == '__main__':
    torch.manual_seed(0)

    # confirm that with_logits works as intended
    fl = FocalLoss(gamma=1, with_logits=True, reduction='none')
    fl2 = FocalLoss(gamma=1, with_logits=False, reduction='none')

    input = torch.randn((5, 10))
    pt = F.softmax(input, dim=-1)
    target = torch.randint(0, 9, (5,))

    print(fl(input, target))
    print(fl2(pt, target))

    # confirm that FocalLoss(gamma=0) is identical to nn.CrossEntropyLoss
    fl_nogamma = FocalLoss(gamma=0, with_logits=True, reduction='mean')
    ce = nn.CrossEntropyLoss(reduction='mean')

    print(fl_nogamma(input, target))
    print(ce(input, target))

    # test focal_loss with ignore_index
    target2 = torch.tensor([1, 8, 5, 1, 9])
    print(focal_loss(input, target2, ignore_index=1, reduction='none'))
    print(focal_loss(input, target2, ignore_index=1, reduction='mean'))