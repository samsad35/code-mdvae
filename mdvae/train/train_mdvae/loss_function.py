from torch import nn
import torch
import numpy as np
import torch.distributions.normal as normal
from torch.autograd import Variable
import torch.distributions.multivariate_normal as multinormal
from scipy.stats import multivariate_normal
import math


class Loss:
    def __init__(self, n_wu: int = None):
        self.mse = nn.MSELoss(reduction='sum')
        self.bce = nn.BCELoss()
        self.n_mu = n_wu
        if n_wu is not None:
            self.wu = np.linspace(0.0, 1.0, num=n_wu)

    @staticmethod
    def divergence_is(x, y):
        """
            The Itakura-saito divergence
        :param x:
        :param y:
        :return:
        """
        y = y + 1e-10
        ret = torch.sum(x / y - torch.log(x / y) - 1)
        return ret

    @staticmethod
    def mse_(x, y):
        """
            Mean Square error
        :param x:
        :param y:
        :return:
        """
        ret = torch.sum((x - y) ** 2)
        return ret

    @staticmethod
    def divergence_kl(inference: tuple = None, prior: tuple = None):
        """
            Kullback-Leibler divergence
        :param inference:
        :param prior:
        :return: loss
        """
        assert len(inference) == len(prior) == 3
        mean, logvar = inference[1], inference[2]
        mean_prior, logvar_prior = prior[1], prior[2]
        loss = -0.5 * torch.sum(logvar - logvar_prior + 1
                                - torch.div(torch.exp(logvar) + (mean - mean_prior).pow(2), torch.exp(logvar_prior)))
        return loss

    @staticmethod
    def loss_MPJPE(x, y, nfeats=2):
        """

        :param x:
        :param y:
        :param nfeats:
        :return:
        """
        bs, seq_len, _ = x.shape
        x = x.reshape(seq_len, bs, -1, nfeats)
        y = y.reshape(seq_len, bs, -1, nfeats)
        ret = (x - y).norm(dim=-1).mean(dim=-1).sum()
        return ret

    def warm_up(self, epoch):
        """

        :param epoch:
        :return:
        """
        a = 1.0
        if epoch < self.n_mu:
            a = self.wu[epoch]
        return a

    def log_density(self, sample, mu, var):
        """

        :param sample:
        :param mu:
        :param var:
        :return:
        """
        c = self.normalization.type_as(sample.data)
        inv_var = torch.exp(-var)
        tmp = (sample - mu) * inv_var
        return -0.5 * (tmp * tmp + 2 * var + c)

    def loss_mdvae(self, x_audio, x_visual, x_audio_recons, x_visual_recons, latent_space,
                   seq_length: int, batch_size: int, beta: float = 1.0):
        r"""

        :param seq_length:
        :param batch_size:
        :param x_audio: Tensor
        :param x_visual:
        :param x_audio_recons:
        :param x_visual_recons:
        :param latent_space:
        :param beta:
        :return:
        """
        loss_zss_kl = self.divergence_kl(latent_space['w'], prior=(0, torch.tensor(0.0), torch.tensor(0.0))) \
                      / batch_size
        loss_zds_kl = self.divergence_kl(latent_space['zav'], prior=latent_space['zav_prior']) \
                      / (batch_size * seq_length)
        loss_z_audio_kl = self.divergence_kl(latent_space['zaudio'], prior=latent_space['zaudio_prior']) \
                          / (batch_size * seq_length)
        loss_z_visual_kl = self.divergence_kl(latent_space['zvisual'], prior=latent_space['zvisual_prior']) \
                           / (batch_size * seq_length)

        loss_audio = self.divergence_is(x_audio, x_audio_recons) / (batch_size * seq_length)
        # loss_audio = self.mse_(x_audio, x_audio_recons) / (self.batch_size * self.seq_length)
        loss_visual = self.mse_(x_visual, x_visual_recons) / (batch_size * seq_length)

        # loss_audio = self.loss_MPJPE(x_audio, x_audio_recons, nfeats=3) / (self.batch_size * self.seq_length)
        # loss_visual = self.loss_MPJPE(x_visual, x_visual_recons, nfeats=3) / (self.batch_size * self.seq_length)

        kl = (loss_zss_kl + loss_zds_kl + loss_z_audio_kl + loss_z_visual_kl)
        reconstruction_loss = (loss_visual + loss_audio)
        total_loss = reconstruction_loss + beta * kl


        return total_loss, dict(kl_Zss=loss_zss_kl, kl_Zaudio=loss_z_audio_kl, kl_Zvisual=loss_z_visual_kl,
                                kl_Zds=loss_zds_kl, loss_audio=loss_audio, loss_visual=loss_visual, kl=kl)

    def __call__(self, *args, **kwargs):
        return self.loss_mdvae(*args, **kwargs)
