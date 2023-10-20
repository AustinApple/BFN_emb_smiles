import torch
import torch.nn as nn
from tqdm.auto import tqdm

class BFNContinuousData(nn.Module):
    def __init__(self, unet, in_channels=1, sigma=0.001):
      super(BFNContinuousData, self).__init__()
      self.sigma = torch.tensor(sigma)
      self.in_channels = in_channels
      self.unet = unet
    def forward(self, theta, t, logp, ema=None):
        """
        Forward pass of the Bayesian Flow Network.

        Parameters
        ----------
        theta : torch.Tensor
            Tensor of shape (B, D).
        t : torch.Tensor
            Tensor of shape (B,).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, D).
        """
        t = self.get_time_embedding(t)
        t += 
        .view(t.shape[0], -1)
        if ema is None:
            output = self.unet(theta, t)
        else:
            with ema.average_parameters():
                output = self.unet(theta, t)
        return output

    def get_time_embedding(self, timestep):
        # timestep: (bs, )
        # output: (bs, 320)
        freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32, device=timestep.device) / 160)
        x = timestep[:, None] * freqs[None]
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

    def cts_output_prediction(self, mu, t, logp, gamma, ema=None, t_min=1e-10, x_range=(-1 , 1)):
        x_pred = self.forward(mu, t, logp, ema=ema)
        mask = t < t_min
        gamma = torch.where(mask[:, None, None], torch.ones_like(gamma), gamma)
        x_pred = mu / gamma - ((1 - gamma) / gamma).sqrt() * x_pred
        x_pred = torch.clamp(x_pred, x_range[0], x_range[1])
        return x_pred

    def process_infinity(self, x, logp, t=None, return_params=False):
        """
        x : torch.Tensor
            Tensor of shape (B, C, H, W).
        t : torch.Tensor
            Tensor of shape (B, ).
        Returns
        -------
        torch.Tensor
            Output tensor of shape (1, ).
        """
        # print(f'x.shape: {x.shape}')
        bs, c, h = x.shape
        
        if t is None:
            t = torch.rand((bs, ), device=x.device)
        gamma = 1 - self.sigma ** (2 * t[:, None, None])  # (B, 1, 1, 1)
        # print(f'gamma: {gamma.shape}')
        mu = gamma * x + (gamma * (1 - gamma)).sqrt() * torch.randn_like(x) # (B, C, H, W)
        # print(f'mu: {mu.shape}')
        pred = self.cts_output_prediction(mu, t, logp, gamma)
        loss_infinity = -torch.log(self.sigma) * self.sigma ** (-2 * t[:, None, None]) * (x - pred)**2
        if return_params:
            return loss_infinity.mean(), mu, pred, t
        else:
            return loss_infinity.mean()
        
    def process_discrete(self, x, step=None, max_step=1000, return_params=False):
        """
        x : torch.Tensor
            Tensor of shape (B, C, H, W).
        t : torch.Tensor
            Tensor of shape (B, ).
        n : int
            number of steps.
        Returns
        -------
        torch.Tensor
            Output tensor of shape (1, ).
        """
        bs, c, h, w = x.shape
        if step is None:
            step = torch.randint(1, max_step + 1, (bs, ), device=x.device)
        t = (step - 1) / max_step
        # t=0　⇨ gamma=0, mu=0, pred=0
        gamma = 1 - self.sigma ** (2 * t[:, None, None, None])
        mu = gamma * x + (gamma * (1 - gamma)).sqrt() * torch.randn_like(x)
        pred = self.cts_output_prediction(mu, t, gamma)
        loss_discrete = max_step * (1 - self.sigma ** (2 / max_step)) / (2 * self.sigma ** (2 * step[:, None, None, None] / max_step)) * (x - pred)**2
        if return_params:
            return loss_discrete.mean(), mu, pred, t
        else:
            return loss_discrete.mean()

    @torch.inference_mode()
    def sample(self, h, logp, batch_size=4, steps=1000, ema=None, return_samples=None, device="cpu"):
        
        self.eval()
        mu = torch.zeros((batch_size, self.in_channels, h), device=device)
        logp = torch.ones((batch_size), device=device) * logp
        rho = 1
        ret_list = []
        for istep in tqdm(range(1, steps+1)):
            t = torch.full((batch_size, ), (istep-1)/steps, device=device)
            x_pred = self.cts_output_prediction(mu, t, logp, 1 - self.sigma ** (2 * t[:, None, None]), ema=ema)
            alpha = self.sigma ** (-2*istep/steps) * (1 - self.sigma ** (2/steps))
            y = x_pred + torch.randn_like(x_pred, device=device) / alpha.sqrt()
            mu = (rho * mu + alpha * y) / (rho + alpha)
            rho += alpha
            if return_samples is not None and (istep-1) % return_samples == 0:
                ret_list.append(x_pred)
                
        t = torch.ones((batch_size, ), device=device)
        x_pred = self.cts_output_prediction(mu, t, logp, 1 - self.sigma ** (2 * t[:, None, None]), ema=ema)
        if return_samples is None:
            return x_pred
        ret_list.append(x_pred)
        return x_pred, ret_list