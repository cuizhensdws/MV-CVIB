
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class MV-CVIB(nn.Module):
    """
    Multimodal Variational Information Bottleneck.
    """
    def __init__(self, n_latents, abundance_dim, SampleNN_dim, device, ):

        super(MV-CVIB, self).__init__()

        self.abundance_encoder = AbundanceEncoder(x_dim=abundance_dim, z_dim=n_latents)
        self.SampleNN_encoder = SampleNNEncoder(x_dim=SampleNN_dim, z_dim=n_latents)

        self.abundance_decoder = GaussianDecoder(x_dim=abundance_dim, z_dim=n_latents)
        self.SampleNN_decoder = BernoulliDecoder(x_dim=SampleNN_dim, z_dim=n_latents)



        self.classifier = nn.Linear(n_latents, 1)
            
        self.experts = ProductOfExperts()

        self.n_latents = n_latents
        self.abundance_dim = abundance_dim
        self.SampleNN_dim= SampleNN_dim
        self.device = device

    def reparametrize(self, mu, logvar):
        """
        Reparameterization trick.
        Samples z from its posterior distribution.
        """
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
          return mu

    def forward(self, abundance=None, SampleNN=None, metabolite=None):
        # infer joint posterior
        mu, logvar = self.infer(abundance, SampleNN)

        # reparametrization trick to sample
        z = self.reparametrize(mu, logvar)

        # autoencoding
        abundance_recon = self.abundance_decoder(z)
        SampleNN_recon = self.SampleNN_decoder(z)

        # classification
        classification_logits = self.classifier(z)

        return abundance_recon, SampleNN_recon, mu, logvar, classification_logits

    def infer(self, abundance=None, SampleNN=None):
        """
        Infer joint posterior q(z|x).
        """
        if abundance is not None:
            batch_size = abundance.size(0)
        elif SampleNN is not None:
            batch_size = SampleNN.size(0)


        # initialize the universal prior expert
        mu, logvar = self.prior_expert((1, batch_size, self.n_latents))
        if abundance is not None:
            a_mu, a_logvar = self.abundance_encoder(abundance)
            mu = torch.cat((mu, a_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, a_logvar.unsqueeze(0)), dim=0)

        if SampleNN is not None:
            m_mu, m_logvar = self.SampleNN_encoder(SampleNN)
            mu = torch.cat((mu, m_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, m_logvar.unsqueeze(0)), dim=0)



        # product of experts to combine gaussians
        mu, logvar = self.experts(mu, logvar)
        return mu, logvar

    def classify(self, abundance=None, SampleNN=None):
        """
        Classification - Compute p(y|x).
        """
        mu, logvar = self.infer(abundance, SampleNN)
        # reparametrization trick to sample
        z = self.reparametrize(mu, logvar)
        classification_logits = self.classifier(z)
        prediction = torch.sigmoid(classification_logits)
        return prediction

    def prior_expert(self, size):
        """
        Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).
        :param size: dimensionality of Gaussian
        """
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.zeros(size))
        mu, logvar = mu.to(self.device), logvar.to(self.device)
        return mu, logvar


class AbundanceEncoder(nn.Module):

    def __init__(self, x_dim, z_dim):
        super(AbundanceEncoder, self).__init__()
        self.x_dim = x_dim
        self.fc1 = nn.Linear(x_dim, x_dim // 2)
        self.fc2 = nn.Linear(x_dim // 2, x_dim // 2)
        self.conv1 = nn.Conv2d(x_dim // 4, x_dim // 4,2)
        self.pool = nn.MaxPool2d(16, 16)
        self.fc31 = nn.Linear(x_dim // 2, z_dim)
        self.fc32 = nn.Linear(x_dim // 2, z_dim)
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        h = self.drop(F.silu(self.fc1(x.view(-1, self.x_dim))))
        h = self.drop(F.silu(self.fc2(h)))
        return self.fc31(h), self.fc32(h)


class SampleNNEncoder(nn.Module):
    """
    Parametrizes q(z|x).
    :param x_dim: input dimension
    :param z_dim: latent dimension
    """
    def __init__(self, x_dim, z_dim):
        super(SampleNNEncoder, self).__init__()
        self.x_dim = x_dim
        self.fc1 = nn.Linear(x_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.conv1 = nn.Conv2d(512,512,2)
        self.pool = nn.MaxPool2d(128, 128)
        self.fc31 = nn.Linear(1024, z_dim)
        self.fc32 = nn.Linear(1024, z_dim)
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        h = self.drop(F.silu(self.fc1(x)))
        h = self.drop(F.silu(self.fc2(h)))
        return self.fc31(h), self.fc32(h)


class BernoulliDecoder(nn.Module):
    """
    Parametrizes p(x|z) for a Bernoulli distribution.
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    https://arxiv.org/abs/1312.6114
    :param z_dim: number of latent dimensions
    :param x_dim: output dimension
    """
    def __init__(self, z_dim, x_dim):
        super(BernoulliDecoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, 512)
        self.fc2 = nn.Linear(512, x_dim)
        self.bn1 = nn.BatchNorm1d(512)

    def forward(self, z):
        h = self.bn1(F.relu(self.fc1(z)))
        return self.fc2(h)


class GaussianDecoder(nn.Module):
    """
    Parametrizes p(x|z) for a Gaussian distribution.
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    https://arxiv.org/abs/1312.6114
    :param z_dim: number of latent dimensions
    :param x_dim: output dimension
    """
    def __init__(self, z_dim, x_dim):
        super(GaussianDecoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, z_dim * 2)
        self.fc2 = nn.Linear(z_dim * 2, x_dim)
        self.fc3 = nn.Linear(z_dim * 2, x_dim)

    def forward(self, z):
        h = torch.tanh(self.fc1(z))
        mu = self.fc2(h)
        sigma = self.fc3(h)
        return mu, sigma


class ProductOfExperts(nn.Module):
    """
    Compute parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.
    """
    def forward(self, mu, logvar, eps=1e-8):
        """
        :param mu: M x D for M experts
        :param logvar: M x D for M experts
        :param eps: constant for stability
        """
        # do computation in double to avoid nans
        mu = mu.double()
        logvar = logvar.double()

        var = torch.exp(logvar) + eps
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)

        # back to float
        pd_mu = pd_mu.float()
        pd_logvar = pd_logvar.float()

        return pd_mu, pd_logvar


class Ensembler:
    """
    A class to ensemble model predictions and latent representations from multiple models.
    """
    def __init__(self):
        self._mu = 0
        self._prediction = 0
        self._count = 0

    def add(self, mu, prediction):
        """
        This method needs to be called to add a model to the ensemble.

        :param mu: Torch tensor of the z Gaussian mean
        :param prediction: Torch tensor with the model predictions on the classification task
        """
        self._mu += mu.cpu().detach().numpy()
        self._prediction += prediction.cpu().detach().numpy()
        self._count += 1

    def avg(self):
        """
        This method averages the models in the ensemble.
        """
        self._mu = self._mu / self._count
        self._prediction = self._prediction / self._count

        return self._mu, self._prediction
