import torch
import torch.nn.functional as F


def fuse(a, b):
    p1 = F.softmax(a, dim=1)
    p2 = F.softmax(b, dim=1)

    c1, pred1 = p1.max(dim=1)
    c2, pred2 = p2.max(dim=1)

    # choose prediction from model with higher confidence
    final_pred = []
    final_prob = []

    for i in range(len(c1)):
        if c1[i] > c2[i]:
            final_pred.append(pred1[i])
            final_prob.append(p1[i])
        else:
            final_pred.append(pred2[i])
            final_prob.append(p2[i])

    return torch.stack(final_prob)
