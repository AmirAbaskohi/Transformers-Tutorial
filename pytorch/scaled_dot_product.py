import math
from torch import nn

class ScaledDotProductAttention(nn.Module):

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        batch_size, head, length, d_tensor = k.size()

        k_t = k.transpose(2, 3)
        score = (q @ k_t) / math.sqrt(d_tensor)

        if mask is not None:
            score = score.masked_fill(mask == 0, -e)

        score = self.softmax(score)
        v = score @ v

        return v, score