import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
import math
from torch.nn import Parameter
import torchaudio

from transformer import EncoderLayer, PositionalEncoding


# linear block
class Liner_Module(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(Liner_Module, self).__init__()
        self.liner = nn.Linear(input_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, input: torch.Tensor):
        x = self.liner(input)
        x = self.bn(x)
        x = self.relu(x)
        return x


# AE
class Auto_encoder(nn.Module):
    def __init__(self, input_dim=640, output_dim=640):
        super(Auto_encoder, self).__init__()
        self.encoder = nn.Sequential(
            Liner_Module(input_dim=input_dim, out_dim=128),
            Liner_Module(input_dim=128, out_dim=128),
            Liner_Module(input_dim=128, out_dim=128),
            Liner_Module(input_dim=128, out_dim=128),
            Liner_Module(input_dim=128, out_dim=8),
        )
        self.decoder = nn.Sequential(
            Liner_Module(input_dim=8, out_dim=128),
            Liner_Module(input_dim=128, out_dim=128),
            Liner_Module(input_dim=128, out_dim=128),
            Liner_Module(input_dim=128, out_dim=128),
            nn.Linear(128, output_dim),
        )

    def forward(self, input: torch.Tensor):
        x_feature = self.encoder(input)
        x = self.decoder(x_feature)
        return x, x_feature


# VAE
class VAE(nn.Module):
    def __init__(self, input_dim=640, output_dim=640):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            Liner_Module(input_dim=input_dim, out_dim=128),
            Liner_Module(input_dim=128, out_dim=128),
            Liner_Module(input_dim=128, out_dim=128),
            Liner_Module(input_dim=128, out_dim=128),
        )
        self.fc_mean = Liner_Module(input_dim=128, out_dim=8)
        self.fc_logvar = Liner_Module(input_dim=128, out_dim=8)
        self.fc_z = Liner_Module(input_dim=128, out_dim=8)

        self.decoder = nn.Sequential(
            Liner_Module(input_dim=8, out_dim=128),
            Liner_Module(input_dim=128, out_dim=128),
            Liner_Module(input_dim=128, out_dim=128),
            Liner_Module(input_dim=128, out_dim=128),
            nn.Linear(128, output_dim),
        )

    # def reparameterization(self, mu, logvar):
    #     std = torch.exp(0.5 * logvar)
    #     rand = torch.randn(std.size()).to(std.device)
    #     z = rand * std + mu
    #     return z

    def reparameterization(self, z, mu, logvar):
        std = torch.exp(0.5 * logvar)
        # rand = torch.randn(std.size()).to(std.device)
        return z * std + mu

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        z = self.fc_z(h)
        z = self.reparameterization(z, mu, logvar)
        output = self.decoder(z)
        if self.training:
            return output, z, mu, logvar
        else:
            return output, z


# Transformer Autoencoder
class TransAE(nn.Module):
    def __init__(self, frames=5, n_enc_layer=2, n_dec_layer=2,
                 n_mel=128, n_fft=513, n_head=4, h_dim=8, n_class=4, dropout=0., lpe=True, cfp=True):
        super(TransAE, self).__init__()
        self.lpe = lpe  # linear phase encoding
        self.cfp = cfp  # center frame prediction
        self.frames = frames - 1 if cfp else frames
        # pe
        self.pos_encoding = PositionalEncoding(d_model=n_mel, seq_len=self.frames, requires_grad=False)
        # lpe
        self.phase_encode = nn.Sequential(
            nn.Linear(n_fft, n_mel),
            nn.BatchNorm1d(self.frames),
            nn.Linear(n_mel, n_mel),
            nn.BatchNorm1d(self.frames),
        )
        # encoder
        self.encoder = nn.ModuleList()
        for _ in range(n_enc_layer):
            encoder = EncoderLayer(n_mel, n_mel, n_head, 4*n_mel, dropout)
            self.encoder.append(encoder)
        # bootleneck-expand
        self.bottleneck = nn.Linear(n_mel, h_dim)
        self.expand = nn.Linear(h_dim, n_mel)
        # decoder
        self.decoder = nn.ModuleList()
        for _ in range(n_dec_layer):
            decoder = EncoderLayer(n_mel, n_mel, n_head, 4*n_mel, dropout)
            self.decoder.append(decoder)
        # reconstruction & classification
        self.fc_rec = nn.Linear(n_mel, n_mel)
        self.fc_clf = self.fc = nn.Sequential(
                                    nn.Linear(h_dim, h_dim),
                                    nn.LayerNorm(h_dim),
                                    nn.ReLU(inplace=True),
                                    # nn.Dropout(0.2),
                                    nn.Linear(h_dim, n_class)
                                )

    def forward(self, input, phase):
        x = input + (self.phase_encode(phase) if self.lpe else self.pos_encoding())
        for enc in self.encoder:
            x, att = enc(x)
        feature = self.bottleneck(x)
        x = self.expand(feature)
        for dec in self.decoder:
            x, att = dec(x)
        if self.cfp:
            x = torch.mean(x, dim=1, keepdim=True)
        output = self.fc_rec(x)
        feat_mean = torch.mean(feature, dim=1)
        feat_max, _ = torch.max(feature, dim=1)
        logit = self.fc_clf(feat_mean + feat_max)
        return output, logit






















