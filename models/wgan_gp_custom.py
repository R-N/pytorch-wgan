import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from .wgan_gradient_penalty import WGAN_GP
from utils.metrics import mile, mire

class WGAN_GP_CUSTOM(WGAN_GP):
    def __init__(self, args, loss_fn=mile):
        super().__init__(args)
        self.loss_fn = loss_fn

    def grad_penalty_loss(self, grad_norm):
        return self.loss_fn(grad_norm)

class WGAN_GP_MILE(WGAN_GP_CUSTOM):
    def __init__(self, args):
        super().__init__(args, mile)

class WGAN_GP_MIRE(WGAN_GP_CUSTOM):
    def __init__(self, args):
        super().__init__(args, mire)
