import torch
from tqdm import tqdm
import torch.nn.functional as F
from src.utils import linear_beta_schedule, nb_linear_beta_schedule


####################################################################################################
# Old_NB_Diffusion
####################################################################################################
class OLDNBDiffusion:
    def __init__(
        self,
        timesteps=1000,
        beta_schedule='linear'
    ):
        self.timesteps = timesteps

        if beta_schedule == 'linear':
            # betas = torch.linspace(1, timesteps,timesteps, dtype=torch.float64)
            betas = nb_linear_beta_schedule(timesteps)
        # elif beta_schedule == 'cosine':
        #     betas = cosine_beta_schedule(timesteps)
        # elif beta_schedule == 'sigmoid':
        #     betas = sigmoid_beta_schedule(timesteps)
        # elif beta_schedule == 'sqrt':
        #     betas = sqrt_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas
        # modify
        # NB distribution
        self.A = 30 # 负二项分布，成功次数
        self.a = torch.tensor(self.A) / self.betas[0] # 比例系数，保证负二项分布为 30
        self.sqrt_a = torch.sqrt(self.a)
        self.forward_noise_coef1 = 1/(self.sqrt_a*(self.betas + 1))
        self.p = 0.5
        self.r_s_t = self.a * betas
        # self.r_bar_s_t = 0.5 * self.a *self.betas * (self.betas+1)
        self.r_bar_s_t = torch.cumsum(self.r_s_t, dim=0)
        self.e_noise_s_t = self.r_bar_s_t * ((1-self.p) / self.p)
        self.v_noise_s_t = self.r_bar_s_t *((1-self.p)/ (self.p ** 2))
        self.std_noise_s_t = torch.sqrt(self.v_noise_s_t)
        self.predict_start_from_noise_coef1 = self.sqrt_a * (self.betas + 1)


        # modify
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            #self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
            # self.betas
            2*(self.betas-1)/(self.betas ** 2)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min =1e-20))

        self.posterior_mean_coef1 = (
            # self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
            1/(self.sqrt_a*(self.betas**2))

        )
        self.posterior_mean_coef2 = (
            # (1.0 - self.alphas_cumprod_prev)
            # * torch.sqrt(self.alphas)
            # / (1.0 - self.alphas_cumprod)
            1-1/(self.betas**2)

        )

    # get the param of given timestep t
    @classmethod
    def _extract(cls, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t)
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    # forward diffusion (using the nice property): q(x_t | x_0)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            r_bar_t = self._extract(self.r_bar_s_t, t, x_start.shape)
            p = self.p
            # noise = torch.distributions.NegativeBinomial(r_bar_t, probs=p).sample(x_start.shape)
            noise = torch.distributions.NegativeBinomial(r_bar_t.squeeze(), probs=p).sample(x_start.shape[1:]).permute([-1, 0, 1, 2])
            e_noise_t = self._extract(self.e_noise_s_t, t, x_start.shape)
            std_noise_t = self._extract(self.std_noise_s_t, t, x_start.shape)
            # 标准化noise
            noise = (noise - e_noise_t) / std_noise_t
            noise = noise
        coef1 = self._extract(self.forward_noise_coef1, t, x_start.shape)
        return coef1 * (x_start + noise)

    # modify
    # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # modify
    # compute x_0 from x_t and pred noise: the reverse of `q_sample`
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            # self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            # self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            (self._extract(self.predict_start_from_noise_coef1, t, x_t.shape) * x_t) - noise
        )

    # modify
    # compute predicted mean and variance of p(x_{t-1} | x_t)
    def p_mean_variance(self, model, x_t, t, clip_denoised=True):
        # predict noise using model
        x_t = x_t.float() # 保证输入的数据是float32
        # importantt important important!!!!!
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
        model_mean, model_variance, _ = self.p_mean_variance(model, x_t, t,
                                                    clip_denoised=clip_denoised)
        noise = torch.randn_like(x_t)
        pred_img = model_mean + torch.sqrt(model_variance).float() * noise
        return pred_img

    # denoise: reverse diffusion
    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        batch_size = shape[0]
        device = next(model.parameters()).device
        r_bar_t = self.r_bar_s_t[-1]
        img = torch.distributions.NegativeBinomial(r_bar_t, probs=self.p).sample(shape).float()
        e_noise_t = self.e_noise_s_t[-1]
        std_noise_t = self.std_noise_s_t[-1]
        # 标准化noise
        img = (img - e_noise_t) / std_noise_t
        img = img.to(device)
        #img = img.reshape([-1, shape[0]]).T
        #img = img.reshape(shape) - self.kappas_cumsum[-1] * self.thetas[-1]
        imgs = []
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(model, img, torch.full((batch_size,), i, device=device, dtype=torch.long))
            imgs.append(img.cpu().numpy())
        return imgs

    # sample new images
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=8, channels=3):
        return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))

    # compute train losses
    def train_losses(self, model, x_start, t):
        r_bar_t = self._extract(self.r_bar_s_t, t, x_start.shape)
        p = self.p
        noise = torch.distributions.NegativeBinomial(r_bar_t.squeeze(), probs=p).sample(x_start.shape[1:]).permute([-1,0,1,2])
        e_noise_t = self._extract(self.e_noise_s_t, t, x_start.shape)
        std_noise_t = self._extract(self.std_noise_s_t, t, x_start.shape)
        # 标准化noise
        noise = (noise - e_noise_t) / std_noise_t
        # get x_t
        x_noisy = self.q_sample(x_start, t, noise=noise)
        predicted_noise = model(x_noisy, t)
        loss = F.mse_loss(noise, predicted_noise)
        return loss