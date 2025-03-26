import torch
from tqdm import tqdm

from src.utils import cosine_beta_schedule, sigmoid_beta_schedule, sqrt_beta_schedule, \
    gaussian_linear_beta_schedule, gaussian_v2_linear_beta_schedule
import torch.nn.functional as F


####################################################################################################
# GaussianDiffusion
####################################################################################################
class GaussianDiffusion:
    def __init__(
            self,
            timesteps=1000,
            beta_schedule='linear'
    ):
        self.timesteps = timesteps

        if beta_schedule == 'linear':
            # betas = gaussian_linear_beta_schedule(timesteps)
            betas = gaussian_v2_linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == 'sigmoid':
            betas = sigmoid_beta_schedule(timesteps)
        elif beta_schedule == 'sqrt':
            betas = sqrt_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas

        self.a = 2215
        self.r=100


        self.t = torch.linspace(1, timesteps, timesteps, dtype=torch.float32)#t是步数，1到1000

        self.r_bar = self.r * self.t
        self.u =  self.r * self.t


        self.betas_cumprod=torch.cumprod(self.betas, axis=0)
        # self.alphas = 1. - self.betas
        self.alphas = 1 * self.betas_cumprod
        # self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod = self.alphas
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        self.sqrt_alphas=torch.sqrt(self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)

        self.sqrt_one_minus_alphas = torch.sqrt(1.0 - self.alphas)

        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # # gamma atribute
        # self.theta_0 = theta_0
        # self.kappas = (self.betas / self.alphas_cumprod / self.theta_0 ** 2)
        # self.thetas = (self.theta_0 * self.sqrt_alphas_cumprod)
        # self.kappas_cumsum = torch.cumsum(self.kappas, axis=0)
        # self.kappas_cumsum_thetas = (self.kappas_cumsum * self.thetas)
        # self.log_betas = torch.log(self.betas)
        # self.kappas_cumsum_thetas_over_sqrt_alphas_cumprod = (
        #             self.sqrt_recip_alphas_cumprod * self.kappas_cumsum_thetas)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
            # self.betas
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

        self.posterior_mean_coef1 = (
                self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * torch.sqrt(self.alphas)
                / (1.0 - self.alphas_cumprod)
        )

    # get the param of given timestep t
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    # forward diffusion (using the nice property): q(x_t | x_0)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            r_bar_t = self._extract(self.r_bar, t, x_start.shape)
            u_t = self._extract(self.u, t, x_start.shape)
            # noise = torch.randn_like(x_start)
            noise = torch.distributions.Poisson(r_bar_t.squeeze()).sample(x_start.shape).permute([-1,0,1,2])
            sqrt_alphas_t = self._extract(self.sqrt_alphas, t, x_start.shape)
            sqrt_one_minus_alphas_t = self._extract(self.sqrt_one_minus_alphas, t, x_start.shape)
            noise = (sqrt_alphas_t * (noise-u_t))/(self.a*sqrt_one_minus_alphas_t)

            # kappas_cumsum_t = self._extract(self.kappas_cumsum, t, x_start.shape)
            # thetas_t = self._extract(self.thetas, t, x_start.shape)
            # noise = Gamma(kappas_cumsum_t.squeeze(), (1 / thetas_t).squeeze()).sample(x_start.shape)

        # kappas_cumsum_thetas_t = self._extract(self.kappas_cumsum_thetas, t, x_start.shape)
        # sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_alphas_t = self._extract(self.sqrt_alphas, t, x_start.shape)
        # sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_t = self._extract(self.sqrt_one_minus_alphas, t, x_start.shape)

        # return sqrt_alphas_cumprod_t * x_start + noise - kappas_cumsum_thetas_t
        # return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return sqrt_alphas_t * x_start + sqrt_one_minus_alphas_t * noise

    # Get the mean and variance of q(x_t | x_0).
    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        # variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        variance = self._extract(self.betas, t, x_start.shape)
        # log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_betas, t, x_start.shape)
        return mean, variance, log_variance

    # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
                self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # compute x_0 from x_t and pred noise: the reverse of `q_sample`
    def predict_start_from_noise(self, x_t, t, noise):
        return (
                self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    # compute predicted mean and variance of p(x_{t-1} | x_t)
    def p_mean_variance(self, model, x_t, t, clip_denoised=True):
        # predict noise using model
        pred_noise = model(x_t, t)
        # get the predicted x_0: different from the algorithm2 in the paper
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1., max=1.)
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance

    # denoise_step: sample x_{t-1} from x_t and pred_noise
    @torch.no_grad()
    def p_sample(self, model, x_t, t, clip_denoised=True):
        # predict mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t,
                                                                 clip_denoised=clip_denoised)
        noise = torch.randn_like(x_t)
        # # gamma noise
        # kappas_cumsum_tm1 = self._extract(self.kappas_cumsum, t, x_t.shape)
        # thetas_tm1 = self._extract(self.thetas, t, x_t.shape)
        # kappas_cumsum_thetas_tm1 = self._extract(self.kappas_cumsum_thetas, t, x_t.shape)
        # sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        # gamma_tm1 = Gamma(kappas_cumsum_tm1.squeeze(), (1 / thetas_tm1.squeeze())).sample(x_t.shape[1:])
        # gamma_tm1 = gamma_tm1.reshape([-1, x_t.shape[0]]).T
        # gamma_tm1 = gamma_tm1.reshape(x_t.shape)
        # noise = (gamma_tm1 - kappas_cumsum_thetas_tm1) / sqrt_one_minus_alphas_cumprod_t

        # no noise when t == 0
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        # compute x_{t-1}
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img

    # denoise: reverse diffusion
    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        batch_size = shape[0]
        device = next(model.parameters()).device
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        # kappas_sumsum_T = self.kappas_cumsum[-1]
        # thetas_T = self.thetas[-1]
        # img = Gamma(kappas_sumsum_T, (1 / thetas_T)).sample(shape).float() - kappas_sumsum_T * thetas_T
        img = img.to(device)
        # img = img.reshape([-1, shape[0]]).T
        # img = img.reshape(shape) - self.kappas_cumsum[-1] * self.thetas[-1]
        imgs = []
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(model, img, torch.full((batch_size,), i, device=device, dtype=torch.long))
            imgs.append(img.cpu().numpy())
        return imgs
    
    @torch.no_grad()
    def p_sample_loopA(self, model,img,shape):
        batch_size = shape[0]
        device = next(model.parameters()).device
        # start from pure noise (for each example in the batch)
        # img = torch.randn(shape, device=device)
        # kappas_sumsum_T = self.kappas_cumsum[-1]
        # thetas_T = self.thetas[-1]
        # img = Gamma(kappas_sumsum_T, (1 / thetas_T)).sample(shape).float() - kappas_sumsum_T * thetas_T
        img = img.to(device)
        # img = img.reshape([-1, shape[0]]).T
        # img = img.reshape(shape) - self.kappas_cumsum[-1] * self.thetas[-1]
        imgs = []
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(model, img, torch.full((batch_size,), i, device=device, dtype=torch.long))
            imgs.append(img.cpu().numpy())
        return imgs

    # sample new images
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=8, channels=3):
        return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))

    @torch.no_grad()
    def sampleA(self, model, image_size, img,batch_size=8, channels=3):
        
        img = self.q_sample(img,torch.full((batch_size,), 999, dtype=torch.long))
        return self.p_sample_loopA(model, img,shape=(batch_size, channels, image_size, image_size))

    # compute train losses
    def train_losses(self, model, x_start, t):
        # generate random noise
        # noise = torch.randn_like(x_start)
        r_bar_t = self._extract(self.r_bar, t, x_start.shape)
        u_t = self._extract(self.u, t, x_start.shape)
        noise = torch.distributions.Poisson(r_bar_t.squeeze()).sample(x_start.shape).permute([-1,0,1,2])
        sqrt_alphas_t = self._extract(self.sqrt_alphas, t, x_start.shape)
        sqrt_one_minus_alphas_t = self._extract(self.sqrt_one_minus_alphas, t, x_start.shape)
        noise = (sqrt_alphas_t * (noise-u_t))/(self.a*sqrt_one_minus_alphas_t)

        # sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        # kappas_cumsum_t = self._extract(self.kappas_cumsum, t, x_start.shape)
        # thetas_t = self._extract(self.thetas, t, x_start.shape)
        # noise = Gamma(kappas_cumsum_t.squeeze(), (1 / thetas_t).squeeze()).sample(x_start.shape[1:])
        noise = noise.reshape([-1, x_start.shape[0]]).T
        noise = noise.reshape(x_start.shape)

        # get x_t
        x_noisy = self.q_sample(x_start, t, noise=noise)
        predicted_noise = model(x_noisy, t)
        # gamma_noise = (noise - kappas_cumsum_t * thetas_t) / sqrt_one_minus_alphas_cumprod_t
        # loss = F.mse_loss(gamma_noise, predicted_noise)
        loss = F.mse_loss(noise, predicted_noise)
        return loss