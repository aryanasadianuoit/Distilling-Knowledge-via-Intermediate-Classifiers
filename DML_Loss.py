from torch import nn
from torch.nn import functional as F


def dml_loss_function( student_peer_output,
                           teacher_peers_outputs,
                           target_labels,
                           alpha = 0.1,
                           temperature = 1,
                           burn_out_mode = False):

    """

    :param student_peer_output: The model that should be optimized
    :param teacher_peers_outputs: the cohort of peers which acts as a group of teacher peers.
    :param target_labels: the ground truth labels for the batch.
    :param alpha: the alpha weight to balance the cross-entropy loss (regular training) and the loss
    for comparing the the probability distribution of the studnt and the cohort of teacher peers. default = 0.1
    :param temperature: default = 1
    :param burn_out_mode: the first few epochs that acts as a warmer for the peers. if True, it just computes the regular
    cross-entropy loss. default = False
    :return: total loss for DML
    """

    teacher_peers_outputs = dict(teacher_peers_outputs)
    cross_entropy_loss = (F.cross_entropy(student_peer_output, target_labels))


    if burn_out_mode == False:
        total_kl_loss = 0.0
        for (key, value) in teacher_peers_outputs.items():
            distillation_loss = (nn.KLDivLoss(reduction="batchmean")
                                 (F.log_softmax(student_peer_output / temperature, dim=1),
                                  F.softmax(value / temperature, dim=1))) * \
                                (alpha * temperature * temperature)
            total_kl_loss += distillation_loss

        total_kl_loss *= (1 / len(teacher_peers_outputs.keys()))

        cross_entropy_loss = cross_entropy_loss* (1. - alpha)

        return cross_entropy_loss + total_kl_loss
    else:
        return cross_entropy_loss
