# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Dict, List, Sequence
from matplotlib import figure
import torch


def accuracy_at_k(
    outputs: torch.Tensor, targets: torch.Tensor, top_k: Sequence[int] = (1, 5)
) -> Sequence[int]:
    """Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        outputs (torch.Tensor): output of a classifier (logits or probabilities).
        targets (torch.Tensor): ground truth labels.
        top_k (Sequence[int], optional): sequence of top k values to compute the accuracy over.
            Defaults to (1, 5).

    Returns:
        Sequence[int]:  accuracies at the desired k.
    """

    with torch.no_grad():
        maxk = max(top_k)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def weighted_mean(outputs: List[Dict], key: str, batch_size_key: str) -> float:
    """Computes the mean of the values of a key weighted by the batch size.

    Args:
        outputs (List[Dict]): list of dicts containing the outputs of a validation step.
        key (str): key of the metric of interest.
        batch_size_key (str): key of batch size values.

    Returns:
        float: weighted mean of the values of a key
    """

    value = 0
    n = 0
    for out in outputs:
        value += out[batch_size_key] * out[key]
        n += out[batch_size_key]
    value = value / n
    return value.squeeze(0)


def tensor_mean(outputs: List[Dict], key: str, batch_size_key: str) -> torch.Tensor:
    """
    Computes the mean tensor of the values of a key. Only tensors of the same batch size are considered.

    Args:
        outputs (List[Dict]): list of dicts containing the outputs of a validation step.
        key (str): key of the tensor metric of interest.
        batch_size_key (str): key of batch size values.

    Returns:
        torch.Tensor: mean tensor of the values of a key
    """
    batch_size = outputs[0][batch_size_key]
    tensors = torch.stack(
        [output[key] for output in outputs if output[batch_size_key] == batch_size]
    )
    return tensors.mean(dim=0)


def get_heatmap(matrix: torch.Tensor, norm: str = None) -> figure.Figure:
    """
    Returns a heatmap of the given matrix.
    
    Args:
        matrix (torch.Tensor): matrix to plot.

    Returns:
        figure.Figure: heatmap of the matrix.
    """
    fig = figure.Figure()
    ax = fig.subplots(1)
    ax.imshow(matrix.detach().cpu().numpy(), norm=norm)
    return fig
