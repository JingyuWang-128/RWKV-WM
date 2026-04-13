import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import modules.networks as net
import modules.functions_losses as func
import utils
import scan


params = lambda x: list(x.parameters())
swap = lambda x: torch.transpose(x, 0, 1)
percentile = lambda x, per: torch.quantile(x, per)
expand = lambda x, n: x[None, ...].expand(n, *[1 for _ in range(x.dim())])


class ActorCriticAgent(nn.Module):
    def __init__(self,
                 num_action,
                 feat,
                 hidden,
                 entropy_coef,
                 num_sample,
                 total_train_steps,
                 num_bin,
                 max_bin,
                 min_per,
                 max_per,
                 ema_decay,
                 gamma,
                 lambd,
                 tau,
                 lr,
                 eps,
                 use_amp,
                 act, 
                 device,
                 ):
        super().__init__()
        self.device = device
        self.num_action = num_action
        self.entropy_coef = entropy_coef
        self.entropy_coef_min = entropy_coef * 0.1
        self.total_train_steps = max(1, int(total_train_steps))
        self.num_sample = num_sample
        self.min_per = min_per
        self.max_per = max_per
        self.gamma = gamma
        self.lambd = lambd
        self.tau = tau
        self.slow_reg_coef = 0.1

        self.device_type = "cuda" if "cuda" in device else "cpu"
        self.tensor_dtype = torch.float16 if use_amp else torch.float32
        self.use_amp = use_amp

        self.actor = net.AgentLayer(feat, num_action, hidden, act)
        self.critic = net.AgentLayer(feat, num_bin, hidden, act)
        self.slow_critic = copy.deepcopy(self.critic)

        self.lower_ema = utils.EMAScalar(ema_decay)
        self.upper_ema = utils.EMAScalar(ema_decay)

        agent_params = params(self.actor) + params(self.critic)
        self.twohot_loss = func.SymLogTwoHotLoss(num_bin, -max_bin, max_bin)
        self.optimizer = torch.optim.AdamW(agent_params, lr=lr, eps=eps)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
    
    @torch.no_grad()
    def update_slow_critic(self):
        for target, model in zip(self.slow_critic.parameters(), self.critic.parameters()):
            target.data.copy_(target.data * (1 - self.tau) + model.data * self.tau)

    @torch.no_grad()
    def sample(self, feat, greedy=False):
        self.eval()
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            dist = Categorical(logits=self.actor(feat))
            action = dist.probs.argmax(dim=-1) if greedy else dist.sample()
        return action

    def sample_as_env_action(self, feat, greedy=False):
        action = self.sample(feat, greedy)
        env_action = action.detach().cpu().numpy()
        return env_action, action

    @torch.no_grad()
    def get_logprob(self, feat, action):
        self.eval()
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            dist = Categorical(logits=self.actor(feat))
            log_prob = dist.log_prob(action)[..., None]
        return log_prob
    
    @torch.no_grad()
    def get_scale(self, samples=None):
        if samples is None:
            lower_bound = self.lower_ema.get()
            upper_bound = self.upper_ema.get()
        else:
            lower_bound = self.lower_ema(percentile(samples, self.min_per))
            upper_bound = self.upper_ema(percentile(samples, self.max_per))
        scale = max(upper_bound - lower_bound, 1.0)
        return scale
    
    def get_logits_raw_value(self, x):
        logits = self.actor(x)
        raw_value = self.critic(x)
        return logits, raw_value

    def _slow_critic_reg_loss(self, logits, slow_logits, weight):
        slow_target = torch.softmax(slow_logits.detach(), dim=-1)
        reg = -slow_target * torch.log_softmax(logits, dim=-1)
        reg = reg.sum(dim=-1, keepdim=True)
        return torch.mean(reg * weight)

    def _get_entropy_coef(self, step):
        if step is None:
            return self.entropy_coef
        anneal_span = max(1.0, self.total_train_steps * 0.3)
        progress = min(1.0, step / anneal_span)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.entropy_coef_min + (self.entropy_coef - self.entropy_coef_min) * cosine
    
    def update(self, feat, action, discount, reward, weight, logger=None, step=None):
        self.train()
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            feat = feat.detach()
            logits, raw_value = self.get_logits_raw_value(feat)
            slow_raw_value = self.slow_critic(feat)

            dist = Categorical(logits=logits[:, :-1])
            log_prob = dist.log_prob(action)[..., None]
            entropy = dist.entropy()[..., None]

            with torch.no_grad():
                slow_value = self.twohot_loss.decode(slow_raw_value)
                swap_rew, swap_val, swap_disc = swap(reward), swap(slow_value), swap(discount)
                lambda_return = swap(scan.parallel_lambda_return(
                    swap_rew, swap_val[:-1], swap_val[1:], swap_disc, self.lambd))
                adv = lambda_return - slow_value[:, :-1]
                scale = self.get_scale(lambda_return)
                adv_std = adv.std().clamp_min(1e-4)
                norm_adv = (adv - adv.mean()) / adv_std

            value_loss = torch.mean(self.twohot_loss(raw_value[:, :-1], lambda_return, reduce=False) * weight)
            slow_reg_loss = self._slow_critic_reg_loss(raw_value[:, :-1], slow_raw_value[:, :-1], weight)
            critic_loss = value_loss + self.slow_reg_coef * slow_reg_loss

            policy_loss = -torch.mean(log_prob * norm_adv.detach() * weight.detach())
            entropy_bonus = torch.mean(entropy * weight.detach())
            curr_entropy_coef = self._get_entropy_coef(step)
            actor_loss = policy_loss - curr_entropy_coef * entropy_bonus

            total_loss = critic_loss + actor_loss

        # gradient descent
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        self.update_slow_critic()

        if logger is not None:
            td_error = lambda_return - slow_value[:, :-1]
            logger.log('ActorCritic/value_loss', value_loss.mean().item(), step)
            logger.log('ActorCritic/slow_reg_loss', slow_reg_loss.mean().item(), step)
            logger.log('ActorCritic/policy_loss', policy_loss.mean().item(), step)
            logger.log('ActorCritic/actor_loss', actor_loss.mean().item(), step)
            logger.log('ActorCritic/entropy', entropy_bonus.mean().item(), step)
            logger.log('ActorCritic/entropy_coef', curr_entropy_coef, step)
            logger.log('ActorCritic/scale', scale, step)
            logger.log('ActorCritic/lambda_return', lambda_return.mean().item(), step)
            logger.log('ActorCritic/lambda_return_std', lambda_return.std().item(), step)
            logger.log('ActorCritic/adv_mean', adv.mean().item(), step)
            logger.log('ActorCritic/adv_abs_mean', adv.abs().mean().item(), step)
            logger.log('ActorCritic/adv_std', adv.std().item(), step)
            logger.log('ActorCritic/norm_adv', norm_adv.mean().item(), step)
            logger.log('ActorCritic/norm_adv_abs_mean', norm_adv.abs().mean().item(), step)
            logger.log('ActorCritic/norm_adv_std', norm_adv.std().item(), step)
            logger.log('ActorCritic/td_error_mean', td_error.mean().item(), step)
            logger.log('ActorCritic/td_error_std', td_error.std().item(), step)
            logger.log('ActorCritic/td_error_abs_mean', td_error.abs().mean().item(), step)
            logger.log('ActorCritic/td_error_p10', torch.quantile(td_error, 0.10).item(), step)
            logger.log('ActorCritic/td_error_p50', torch.quantile(td_error, 0.50).item(), step)
            logger.log('ActorCritic/td_error_p90', torch.quantile(td_error, 0.90).item(), step)
            logger.log('ActorCritic/policy_to_entropy_ratio',
                       abs(policy_loss.mean().item()) / max(curr_entropy_coef * entropy_bonus.mean().item(), 1e-8),
                       step)
