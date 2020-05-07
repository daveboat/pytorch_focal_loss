# Focal Loss in Pytorch

**May, 2020: See my updated implementation in [github.com/daveboat/dy_common](https://github.com/daveboat/dy_common/blob/master/dy_common/pytorch_helpers/focalloss.py),
which adds support for label smoothing.** 
__________________________________________________________________

This is a pytorch implmentation of focal loss (https://arxiv.org/abs/1708.02002), meant to be understandable and easily swappable with nn.CrossEntropyLoss and F.cross_entropy. This implementation includes the `ignore_index` parameter like pytorch's cross entropy functions.

Everything is in `focalloss.py` and should be well-commented enough to be self-explanatory. You can either use  the regular function version or the `nn.Module`-wrapped version.