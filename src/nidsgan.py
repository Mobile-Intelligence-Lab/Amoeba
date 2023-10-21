import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from constraints import *


class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Linear(200, output_dim),
        )

    def forward(self, x):
        shape = x.shape
        x = self.layers(x.view(shape[0], -1))
        return x.view(shape) / x.abs().max()


class Discriminator(nn.Module):
    def __init__(self, n_input_features=166, n_classes=2):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_input_features, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class NIDSGAN(nn.Module):
    def __init__(self, classifier, device, input_size, lr=0.00005, clip_value=0.01, n_critic=5,
                 use_network_constraints=False, clip_min=0., clip_max=1., model_save_path=None, params=None, **kwargs):
        super(NIDSGAN, self).__init__()

        classifier = classifier.to(device)

        self.classifier = classifier
        self.lr = lr
        self.n_critic = n_critic
        self.clip_value = clip_value
        self.device = device
        self.use_network_constraints = use_network_constraints
        self.model_save_path = model_save_path
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.params = params

        self.constraints = Constraints(size_indices=SIZE_INDICES, time_indices=TIME_INDICES,
                                       mif_params=MIF_PARAMS, correlated_features=CORRELATED_FEATURES,
                                       max_eps=MAX_EPS, max_eps_time=MAX_EPS_TIME)

        self.generator = Generator(input_size, input_size).to(device)
        self.discriminator = Discriminator(input_size).to(device)

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(.5, .9))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(.5, .9))

        self.CE_Loss = nn.CrossEntropyLoss()
        self.best_loss = float('inf')
        self.best_state = None

    def train_model(self, dataloader, num_epochs):
        # Freeze classifier
        for p in self.classifier.parameters():
            p.requires_grad = False

        for epoch in range(num_epochs):
            epoch_loss = 0
            for i, data in enumerate(dataloader):
                real_samples, real_labels = data[0], data[-1]
                real_samples = torch.tensor(real_samples, dtype=torch.float32).to(self.device)
                real_labels = real_labels.to(self.device).long()

                xs = [real_samples]
                if real_samples.ndim == 3:
                    xs.append(data[1].to(self.device))

                self.generator.eval()
                self.discriminator.train()

                loss_D = 0
                adv_samples, delta = self.generate_adv_samples(real_samples)
                for _ in range(self.n_critic):
                    self.optimizer_D.zero_grad()

                    d_real = self.discriminator(real_samples.view(real_samples.shape[0], -1))
                    d_adv = self.discriminator(adv_samples.view(real_samples.shape[0], -1).detach())
                    loss = F.cross_entropy(d_real, torch.zeros_like(real_labels).long())
                    loss += F.cross_entropy(d_adv, torch.ones_like(real_labels).long())
                    loss_D = loss.item()
                    loss.backward()

                    self.optimizer_D.step()

                # Train generator
                self.generator.train()
                self.discriminator.eval()

                self.optimizer_G.zero_grad()

                # Generate adv samples
                adv_samples, delta = self.generate_adv_samples(real_samples.detach())

                if real_samples.ndim == 3:
                    classifier_outputs = self.classifier(*[adv_samples, xs[1]])
                else:
                    classifier_outputs = self.classifier(adv_samples)
                d_outputs = self.discriminator(adv_samples.view(real_samples.shape[0], -1))

                loss_adv = F.cross_entropy(classifier_outputs, 1 - real_labels)
                loss_discr = F.cross_entropy(d_outputs, torch.zeros_like(real_labels))

                if real_samples.ndim == 3:
                    loss_pert = (torch.clamp(delta[:, :, 0].norm(2, dim=1) - MAX_EPS, min=0.)).mean()
                    loss_pert += (torch.clamp(delta[:, :, 1].norm(2, dim=1) - MAX_EPS_TIME, min=0.)).mean()
                    loss_pert += torch.clamp(delta.norm(2, dim=1) - MAX_EPS, min=0.).mean()
                else:
                    all_correlated_features = list(CORRELATED_FEATURES[0]) + list(CORRELATED_FEATURES[1])
                    non_correlated_size_indices = list(set(SIZE_INDICES) - set(all_correlated_features))
                    non_correlated_time_indices = list(set(TIME_INDICES) - set(all_correlated_features))

                    loss_pert = (
                        torch.clamp(delta[:, non_correlated_size_indices].norm(2, dim=1) - MAX_EPS, min=0.)).mean()
                    loss_pert += (
                        torch.clamp(delta[:, non_correlated_time_indices].norm(2, dim=1) - MAX_EPS_TIME, min=0.)).mean()
                    loss_pert += torch.clamp(delta.norm(2, dim=1) - MAX_EPS, min=0.).mean()

                # Compute Generator loss
                loss_G = loss_adv + .1 * loss_discr + 1 * loss_pert
                epoch_loss += loss_G.item()
                loss_G.backward()
                self.optimizer_G.step()

                # Print loss
                if i % 10 == 0:
                    print(f"[{epoch + 1: 3d}/{num_epochs}]|[{i}/{len(dataloader)}] \t"
                          f"Losses: G={loss_G.item():.4f} | D={loss_D:.4f}")

            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.best_state = self.state_dict()

            if self.model_save_path is not None:
                torch.save(self, self.model_save_path)

        self.load_state_dict(self.best_state)

        if self.model_save_path is not None:
            torch.save(self, self.model_save_path)

    def generate_adv_samples(self, x):
        delta = self.generator(x)

        if self.use_network_constraints and x.ndim < 3:
            delta = self.constraints.enforce_delta_constraints(delta, min_delta=0.)

        adv_samples = x + delta

        if self.use_network_constraints:
            adv_constraints = self.constraints.enforce_adv_constraints if adv_samples.ndim < 3 else \
                self.constraints.enforce_rnn_adv_constraints
            adv_samples = adv_constraints(x, adv_samples, self.clip_min, self.clip_max)

        adv_samples = torch.clamp(adv_samples, min=self.clip_min, max=self.clip_max)

        return adv_samples, delta

    def forward(self, xs):
        outputs = self.classifier(*xs)
        return outputs
