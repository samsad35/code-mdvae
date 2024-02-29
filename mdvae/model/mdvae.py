#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Software mdvae-multimodal dynamical variational auto-encoder
    Copyright CentraleSupelec
    Year November 2021
    Contact : samir.sadok@centralesupelec.fr
"""

from torch import nn
import torch
from collections import OrderedDict
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.nn import Sequential, GRU


class MDVAE2stages(nn.Module):
    bi_rnn_zss: GRU
    audio_features: Sequential
    visual_features: Sequential
    mlp_zss: Sequential
    mlp_zds: Sequential

    def __init__(self, config_model: dict = None, vqvae_speech=None, vqvae_visual=None):
        super(MDVAE2stages, self).__init__()
        self.config_model = config_model
        self.dim_x_audio = self.config_model['dim_x_audio']
        self.dim_x_visual = self.config_model['dim_x_visual']
        self.dim_zss = self.config_model['dim_zss']
        self.dim_zds = self.config_model['dim_zds']
        self.dim_z_audio = self.config_model['dim_z_audio']
        self.dim_z_visual = self.config_model['dim_z_visual']
        self.device = torch.device(self.config_model['device'])
        self.build()

        # == ==
        self.audio_vqvae = vqvae_speech.to(self.device)
        # == ==
        self.vq = vqvae_visual.to(self.device)

    def bloc_audio_features(self):
        dic_layers = OrderedDict()
        dic_layers['Layer1_A'] = nn.Identity()
        self.audio_features = nn.Sequential(dic_layers)

    def bloc_visual_features(self):
        dic_layers = OrderedDict()
        # CNN2D : 1
        dic_layers['Layer1_V_en'] = nn.Linear(32 * 8 * 8, 1024)
        dic_layers['Activation1_V_en'] = nn.ReLU()
        dic_layers['Layer2_V_en'] = nn.Linear(1024, 512)
        dic_layers['Activation2_V_en'] = nn.ReLU()
        self.visual_features = nn.Sequential(dic_layers)

    def bloc_zss(self):
        """

        :return:
        """
        ''' 1) The inference of zss'''
        self.bi_rnn_zss = nn.GRU(input_size=512 + 512, hidden_size=256, num_layers=1,
                                 bidirectional=True, batch_first=True)
        dic_layers = OrderedDict()
        dic_layers['Linear1_zss'] = nn.Linear(2 * 256, 256)
        dic_layers['activation1_Zss'] = nn.Tanh()
        self.mlp_zss = nn.Sequential(dic_layers)
        self.inf_zss_mean = nn.Linear(256, self.dim_zss)
        self.inf_zss_logvar = nn.Linear(256, self.dim_zss)

    def bloc_zds(self):
        """

        :return:
        """
        '''1) the prior on zds'''
        self.rnn_zds = nn.GRU(input_size=self.dim_zds, hidden_size=128, num_layers=1, bidirectional=False,
                              batch_first=True)
        dic_layers = OrderedDict()
        dic_layers['Linear1_zds_prior'] = nn.Linear(128, 64)
        dic_layers['activation1_Zds_prior'] = nn.Tanh()
        self.mlp_zds_prior = nn.Sequential(dic_layers)
        self.inf_zds_mean_prior = nn.Linear(64, self.dim_zds)
        self.inf_zds_logvar_prior = nn.Linear(64, self.dim_zds)

        '''2) the inference on zds'''
        dic_layers = OrderedDict()
        dic_layers['Linear1_zds'] = nn.Linear(512 + 128 + 512 + self.dim_zss, 256)
        dic_layers['activation1_Zds_inf'] = nn.Tanh()
        dic_layers['Linear2_zds'] = nn.Linear(256, 128)
        dic_layers['activation2_Zds_inf'] = nn.Tanh()
        self.mlp_zds = nn.Sequential(dic_layers)
        self.inf_zds_mean = nn.Linear(128, self.dim_zds)
        self.inf_zds_logvar = nn.Linear(128, self.dim_zds)

    def bloc_zaudio_zvisual(self):
        """

        :return:
        """
        '''1) the prior on z_audio'''
        self.rnn_z_audio = nn.GRU(input_size=self.dim_z_audio, hidden_size=128, num_layers=1, bidirectional=False,
                                  batch_first=True)

        dic_layers = OrderedDict()
        dic_layers['Linear1_zaudio_prior'] = nn.Linear(128, 32)
        dic_layers['activation1_Zaudio'] = nn.Tanh()
        self.mlp_zaudio_prior = nn.Sequential(dic_layers)
        self.inf_zaudio_prior_mean = nn.Linear(32, self.dim_z_audio)
        self.inf_zaudio_prior_logvar = nn.Linear(32, self.dim_z_audio)

        '''1) the prior on z_visual'''
        self.rnn_z_visual = nn.GRU(input_size=self.dim_z_visual, hidden_size=128, num_layers=1, bidirectional=False,
                                   batch_first=True)
        dic_layers = OrderedDict()
        dic_layers['Linear1_zvisual_prior'] = nn.Linear(128, 64)
        dic_layers['activation1_Zvisual_prior'] = nn.Tanh()
        self.mlp_zvisual_prior = nn.Sequential(dic_layers)
        self.inf_zvisual_prior_mean = nn.Linear(64, self.dim_z_visual)
        self.inf_zvisual_prior_logvar = nn.Linear(64, self.dim_z_visual)

        '''2) the inference on z_audio'''
        dic_layers = OrderedDict()
        dic_layers['Linear1_zaudio'] = nn.Linear(512 + 128 + self.dim_zss + self.dim_zds, 128)
        dic_layers['activation1_zaudio'] = nn.Tanh()
        dic_layers['Linear2_zaudio'] = nn.Linear(128, 32)
        dic_layers['activation2_zaudio'] = nn.Tanh()
        self.mlp_zaudio = nn.Sequential(dic_layers)
        self.inf_zaudio_mean = nn.Linear(32, self.dim_z_audio)
        self.inf_zaudio_logvar = nn.Linear(32, self.dim_z_audio)

        '''2) the inference on z_visual'''
        dic_layers = OrderedDict()
        dic_layers['Linear1_zvisual'] = nn.Linear(512 + 128 + self.dim_zss + self.dim_zds, 256)
        dic_layers['activation1_zvisual'] = nn.Tanh()
        dic_layers['Linear2_zvisual'] = nn.Linear(256, 128)
        dic_layers['activation2_zvisual'] = nn.Tanh()
        self.mlp_zvisual = nn.Sequential(dic_layers)
        self.inf_zvisual_mean = nn.Linear(128, self.dim_z_visual)
        self.inf_zvisual_logvar = nn.Linear(128, self.dim_z_visual)

    def bloc_x1_x2(self):
        """

        :return:
        """
        '''1) Decoder --> audio'''
        dic_layers = OrderedDict()
        dic_layers['Linear1_DecA'] = nn.Linear(self.dim_z_audio + self.dim_zss + self.dim_zds, 128)
        dic_layers['activation1_zaudio'] = nn.Tanh()
        dic_layers['Linear2_DecA'] = nn.Linear(128, 256)
        dic_layers['activation2_zaudio'] = nn.Tanh()
        dic_layers['Linear3_DecA'] = nn.Linear(256, 512)
        self.de_mlp = nn.Sequential(dic_layers)

        '''2) Decoder --> visual'''
        dic_layers = OrderedDict()
        dic_layers['Layer1_V_dec'] = nn.Linear(self.dim_z_visual + self.dim_zss + self.dim_zds, 512)
        dic_layers['Activation1_V_dec'] = nn.ReLU()
        dic_layers['Layer2_V_dec'] = nn.Linear(512, 1024)
        dic_layers['activation2_zaudio'] = nn.ReLU()
        dic_layers['Linear3_DecA'] = nn.Linear(1024, 2048)
        self.cnn_transpose = nn.Sequential(dic_layers)

    def build(self):
        self.bloc_visual_features()
        self.bloc_audio_features()
        self.bloc_zss()
        self.bloc_zds()
        self.bloc_zaudio_zvisual()
        self.bloc_x1_x2()

    def features(self, x_audio, x_visual, batch_size, seq_lenth):
        features_audio = self.audio_features(x_audio)
        features_visual = self.visual_features(x_visual)
        concatenation_audio_visual_features = torch.cat((features_visual, features_audio), dim=-1)
        return features_audio, features_visual, concatenation_audio_visual_features

    def inference_w(self, input, batch_size):
        h0 = torch.zeros(2, batch_size, 256).to(self.device)
        _, h_w = self.bi_rnn_zss(input.to(self.device), h0)
        h_w = torch.cat((h_w[0, :, :], h_w[1, :, :]), -1)
        zw_ = self.mlp_zss(h_w)
        zw_mean = self.inf_zss_mean(zw_)
        zw_logvar = self.inf_zss_logvar(zw_)
        z_w = self.reparameterization(zw_mean, zw_logvar)
        return z_w, zw_mean, zw_logvar

    def inference_and_prior_zav(self, features_audio, features_visual, z_ss, batch_size, seq_lenth):
        h_n = torch.zeros(1, batch_size, 128).to(self.device)  # initialisation of h_n
        zds_all = torch.tensor([]).to(self.device)
        mean_zds_all = torch.tensor([]).to(self.device)
        logvar_zds_all = torch.tensor([]).to(self.device)
        prior_zds_all = torch.tensor([]).to(self.device)
        prior_mean_zds_all = torch.tensor([]).to(self.device)
        prior_logvar_zds_all = torch.tensor([]).to(self.device)
        for n in range(0, seq_lenth):
            x_audio_n = features_audio[:, n, :]
            x_visual_n = features_visual[:, n, :]
            conditional_zds = torch.cat((h_n[0], z_ss, x_audio_n, x_visual_n), dim=-1)
            zds_n = self.mlp_zds(conditional_zds)
            mean_zds_n = self.inf_zds_mean(zds_n)
            logvar_zds_n = self.inf_zds_logvar(zds_n)
            zds_n = self.reparameterization(mean_zds_n, logvar_zds_n)

            prior_zds_ = self.mlp_zds_prior(h_n[0])
            prior_mean_zds = self.inf_zds_mean_prior(prior_zds_)
            prior_logvar_zds = self.inf_zds_logvar_prior(prior_zds_)
            prior_zds = self.reparameterization(prior_mean_zds, prior_logvar_zds)

            _, h_n = self.rnn_zds(zds_n.unsqueeze(1), h_n)

            # Save the entire sequence :
            zds_all = torch.cat((zds_all, zds_n.unsqueeze(1)), dim=1)
            mean_zds_all = torch.cat((mean_zds_all, mean_zds_n.unsqueeze(1)), dim=1)
            logvar_zds_all = torch.cat((logvar_zds_all, logvar_zds_n.unsqueeze(1)), dim=1)

            prior_zds_all = torch.cat((prior_zds_all, prior_zds.unsqueeze(1)), dim=1)
            prior_mean_zds_all = torch.cat((prior_mean_zds_all, prior_mean_zds.unsqueeze(1)), dim=1)
            prior_logvar_zds_all = torch.cat((prior_logvar_zds_all, prior_logvar_zds.unsqueeze(1)), dim=1)
        return (zds_all, mean_zds_all, logvar_zds_all), (prior_zds_all, prior_mean_zds_all, prior_logvar_zds_all)

    def inference_and_prior_zaudio(self, features_audio, z_ds, z_ss, batch_size, seq_lenth):
        h_n = torch.zeros(1, batch_size, 128).to(self.device)  # initialisation of h_n
        zaudio_all = torch.tensor([]).to(self.device)
        mean_zaudio_all = torch.tensor([]).to(self.device)
        logvar_zaudio_all = torch.tensor([]).to(self.device)
        prior_zaudio_all = torch.tensor([]).to(self.device)
        prior_mean_zaudio_all = torch.tensor([]).to(self.device)
        prior_logvar_zaudio_all = torch.tensor([]).to(self.device)
        for n in range(0, seq_lenth):
            x_audio_n = features_audio[:, n, :]
            z_ds_n = z_ds[:, n, :]
            conditional_zaudio = torch.cat((h_n[0], z_ss, x_audio_n, z_ds_n), dim=-1)
            zaudio_n = self.mlp_zaudio(conditional_zaudio)
            mean_zaudio_n = self.inf_zaudio_mean(zaudio_n)
            logvar_zaudio_n = self.inf_zaudio_logvar(zaudio_n)
            zaudio_n = self.reparameterization(mean_zaudio_n, logvar_zaudio_n)

            prior_zaudio_ = self.mlp_zaudio_prior(h_n[0])
            prior_mean_zaudio = self.inf_zaudio_prior_mean(prior_zaudio_)
            prior_logvar_zaudio = self.inf_zaudio_prior_logvar(prior_zaudio_)
            prior_zaudio = self.reparameterization(prior_mean_zaudio, prior_logvar_zaudio)

            _, h_n = self.rnn_z_audio(zaudio_n.unsqueeze(1), h_n)

            # Save the entire sequence :
            zaudio_all = torch.cat((zaudio_all, zaudio_n.unsqueeze(1)), dim=1)
            mean_zaudio_all = torch.cat((mean_zaudio_all, mean_zaudio_n.unsqueeze(1)), dim=1)
            logvar_zaudio_all = torch.cat((logvar_zaudio_all, logvar_zaudio_n.unsqueeze(1)), dim=1)

            prior_zaudio_all = torch.cat((prior_zaudio_all, prior_zaudio.unsqueeze(1)), dim=1)
            prior_mean_zaudio_all = torch.cat((prior_mean_zaudio_all, prior_mean_zaudio.unsqueeze(1)), dim=1)
            prior_logvar_zaudio_all = torch.cat((prior_logvar_zaudio_all, prior_logvar_zaudio.unsqueeze(1)), dim=1)
        return (zaudio_all, mean_zaudio_all, logvar_zaudio_all), \
               (prior_zaudio_all, prior_mean_zaudio_all, prior_logvar_zaudio_all)

    def inference_and_prior_zvisual(self, features_visual, z_ds, z_ss, batch_size, seq_lenth):
        h_n = torch.zeros(1, batch_size, 128).to(self.device)  # initialisation of h_n
        zvisual_all = torch.tensor([]).to(self.device)
        mean_zvisual_all = torch.tensor([]).to(self.device)
        logvar_zvisual_all = torch.tensor([]).to(self.device)
        prior_zvisual_all = torch.tensor([]).to(self.device)
        prior_mean_zvisual_all = torch.tensor([]).to(self.device)
        prior_logvar_zvisual_all = torch.tensor([]).to(self.device)
        for n in range(0, seq_lenth):
            x_visual_n = features_visual[:, n, :]
            z_ds_n = z_ds[:, n, :]
            conditional_zvisual = torch.cat((h_n[0], z_ss, x_visual_n, z_ds_n), dim=-1)
            zvisual_n = self.mlp_zvisual(conditional_zvisual)
            mean_zvisual_n = self.inf_zvisual_mean(zvisual_n)
            logvar_zvisual_n = self.inf_zvisual_logvar(zvisual_n)
            zvisual_n = self.reparameterization(mean_zvisual_n, logvar_zvisual_n)

            prior_zaudio_ = self.mlp_zvisual_prior(h_n[0])
            prior_mean_zaudio = self.inf_zvisual_prior_mean(prior_zaudio_)
            prior_logvar_zaudio = self.inf_zvisual_prior_logvar(prior_zaudio_)
            prior_zaudio = self.reparameterization(prior_mean_zaudio, prior_logvar_zaudio)

            _, h_n = self.rnn_z_visual(zvisual_n.unsqueeze(1), h_n)

            # Save the entire sequence :
            zvisual_all = torch.cat((zvisual_all, zvisual_n.unsqueeze(1)), dim=1)
            mean_zvisual_all = torch.cat((mean_zvisual_all, mean_zvisual_n.unsqueeze(1)), dim=1)
            logvar_zvisual_all = torch.cat((logvar_zvisual_all, logvar_zvisual_n.unsqueeze(1)), dim=1)

            prior_zvisual_all = torch.cat((prior_zvisual_all, prior_zaudio.unsqueeze(1)), dim=1)
            prior_mean_zvisual_all = torch.cat((prior_mean_zvisual_all, prior_mean_zaudio.unsqueeze(1)), dim=1)
            prior_logvar_zvisual_all = torch.cat((prior_logvar_zvisual_all, prior_logvar_zaudio.unsqueeze(1)), dim=1)
        return (zvisual_all, mean_zvisual_all, logvar_zvisual_all), \
               (prior_zvisual_all, prior_mean_zvisual_all, prior_logvar_zvisual_all)

    def encoder(self, x_audio, x_visual):
        """

        :return:
        """
        batch_size = x_audio.shape[0]
        seq_length = x_audio.shape[1]

        ''' 1)  Features '''
        features_audio, features_visual, cat_audio_visual = self.features(x_audio, x_visual, batch_size, seq_length)

        ''' 2) infer $w$ '''
        w, w_mean, w_logvar = self.inference_w(cat_audio_visual, batch_size)

        ''' 3) infer and dynamic prior $z^{av}$ '''
        inference_zav, prior_zav = self.inference_and_prior_zav(features_audio, features_visual,
                                                                w, batch_size, seq_length)

        ''' 4) infer and dynamic prior z audio '''
        inference_zaudio, prior_zaudio = self.inference_and_prior_zaudio(features_audio, inference_zav[0], w,
                                                                         batch_size, seq_length)

        ''' 5) infer and dynamic prior z visual '''
        inference_zvisual, prior_zvisual = self.inference_and_prior_zvisual(features_visual, inference_zav[0], w,
                                                                            batch_size, seq_length)
        return {"w": (w, w_mean, w_logvar),
                "zav": inference_zav, "zav_prior": prior_zav,
                "zaudio": inference_zaudio, "zaudio_prior": prior_zaudio,
                "zvisual": inference_zvisual, "zvisual_prior": prior_zvisual}

    def decoder(self, zaudio, zvisual, w, zav):
        """

        :return:
        """
        ''' 1) Reconstruction af audio '''
        batch_size = zaudio.shape[0]
        seq_lenth = zaudio.shape[1]
        zw_expand = w.unsqueeze(1).expand(-1, seq_lenth, -1)
        input_audio_decoder = torch.cat((zaudio, zw_expand, zav), dim=-1)
        x_audio_recons = self.de_mlp(input_audio_decoder)
        # x_audio_recons = torch.exp(x_audio_recons)

        ''' 2) Reconstruction af visual '''
        input_visual_decoder = torch.cat((zvisual, zw_expand, zav), dim=-1)
        x_visual_recons = self.cnn_transpose(input_visual_decoder)
        return x_audio_recons, x_visual_recons

    @staticmethod
    def reparameterization(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self, x_audio, x_visual):
        latent_space = self.encoder(x_audio, x_visual)
        x_audio_recons, x_visual_recons = self.decoder(latent_space['zaudio'][0], latent_space['zvisual'][0],
                                                       latent_space['w'][0], latent_space['zav'][0])
        return latent_space, x_audio_recons, x_visual_recons

    def load_model(self, path_model: str):
        checkpoint = torch.load(path_model)
        self.load_state_dict(checkpoint['model'])
        loss = checkpoint['loss']
        print(f"\t - Model VQ-MDVAE is loaded successfully with loss = {loss} ... OK")

    def audio_reconstruction(self, inputs):
        # inputs = torch.reshape(inputs, (inputs.shape[0]*inputs.shape[1], -1))
        out = self.vae_audio.decoder(inputs)
        return out

    def audio_reconstruction_vqvae(self, inputs):
        vq_output_eval = torch.reshape(inputs, (-1, 8, 64))
        loss, quantized, perplexity, _ = self.audio_vqvae._vq_vae(vq_output_eval)
        valid_reconstructions = self.audio_vqvae._decoder(quantized)
        return np.sqrt(torch.transpose(valid_reconstructions[:, 0, :], 0, 1).cpu().detach().numpy())

    def visual_reconstruction(self, indices, path_to_save: str = "", add: str = "", nrow: int = 6, format: str = "png"):
        vq_output_eval = torch.reshape(indices, (-1, 32, 8, 8))
        loss, quantized, perplexity, _ = self.vq._vq_vae(vq_output_eval)
        valid_reconstructions = self.vq._decoder(quantized)

        def show(img, path: str):
            # fig, ax = plt.subplots(figsize=(9, 7))
            fig, ax = plt.subplots(figsize=(15, 15))
            npimg = img.cpu().detach().numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            # plt.show()
            # plt.savefig(f"temps/image_{add}.{format}")
            plt.savefig(f"{path}/{add}.{format}",
                        transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()

        show(make_grid(valid_reconstructions.cpu().detach().data, nrow=nrow) + 0.5, path=path_to_save)
        return valid_reconstructions


# ======================================================================================================================
if __name__ == '__main__':
    mdvae = MDVAE2stages(
        config_model=dict(dim_x_audio=513, dim_x_visual=(1, 64, 64), dim_zss=64, dim_zds=32, dim_z_audio=8,
                          dim_z_visual=32, device='cuda'))
    print(sum(p.numel() for p in mdvae.parameters() if p.requires_grad))
    # print(mdvae)
    torch.cuda.empty_cache()
    mdvae.cuda()
    x_audio = torch.randn(16, 50, 513).to('cuda')
    x_visual = torch.randn(16, 50, 512).to('cuda')
    latent_space, x_audio_recons, x_visual_recons = mdvae(x_audio, x_visual)
    print(x_visual_recons.shape)

