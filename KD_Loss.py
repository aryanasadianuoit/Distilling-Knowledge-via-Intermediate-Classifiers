from torch import nn
from torch.nn import functional as F
criterion = nn.CrossEntropyLoss()


def kd_loss( out_s, out_t, target,alpha ,temperature):

    kd_div_loss = (nn.KLDivLoss(reduction="batchmean")(F.log_softmax(out_s / temperature, dim=1),F.softmax(out_t / temperature, dim=1))) \
                  * (alpha * temperature * temperature)

    Cross_entropy_loss = (F.cross_entropy(out_s, target)) * (1. - alpha)

    total_loss =  kd_div_loss + Cross_entropy_loss

    return total_loss