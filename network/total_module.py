import torch

from torch import nn

from my_utils import poisson_noise_generator
from network.math_module import P_Update, Q_Low_Update, Q_Norm_Update
from network.illumination_low_prox import Q_Low_ProxNet
from network.init_decom import InitialDecomposer
from network.reflection_prox import P_ProxNet


class RetinexUnfoldingNetUnsupervised(nn.Module):
    def __init__(self, num_steps: int = 3, init_lambda=0.2, gamma=0.8):
        super().__init__()
        self.num_steps = num_steps
        self.gamma = gamma

        self.decomposer = InitialDecomposer()

        self.gradient_steps = nn.ModuleList([
            nn.ModuleDict({
                'p': P_Update(init_lambda),
                'q_low': Q_Low_Update(init_lambda),
                'q_norm': Q_Norm_Update(init_lambda),
                'prox_r': P_ProxNet(),
                'prox_l_l': Q_Low_ProxNet(),
                'prox_l_n': Q_Low_ProxNet(),
            }) for i in range(num_steps)
        ])

        self._init_weights()

    def forward(self, I_l):

        if self.training:
            I_n = torch.pow(I_l, self.gamma)
            noisy1 = poisson_noise_generator(I_l)
            noisy2 = poisson_noise_generator(I_n)
            R_l, L_l = self.decomposer(noisy1)
            R_n, L_n = self.decomposer(noisy2)
        else:
            I_n = torch.pow(I_l, self.gamma)
            R_l, L_l = self.decomposer(I_l)
            R_n, L_n = self.decomposer(I_n)

        # I_n = I_l + 0.8 * (I_l - torch.pow(I_l, 2))
        # I_n = torch.pow(I_l, 0.8)
        # 初始分解
        # R_l, L_l = self.decomposer(I_l)
        # R_n, L_n = self.decomposer(I_n)

        state = {'R': R_l, 'L_l': L_l, 'L_n': L_n}

        for step in range(self.num_steps):
            state = self._grad_step(state, I_l, I_n, step)

        return {
            'R_optimized': state['R'],
            'L_n_optimized': state['L_n'],
            'L_l_optimized': state['L_l'],
            'I_enhance': state['R']
        }

    def _grad_step(self, state, I_l, I_n, step):
        modules = self.gradient_steps[step]
        p_k = modules['p'](state['R'], state['L_l'], I_l, state['L_n'], I_n)
        R_k = modules['prox_r'](p_k)
        q_l_k = modules['q_low'](state['L_l'], R_k, I_l)
        q_n_k = modules['q_norm'](state['L_n'], R_k, I_n)
        L_l_k = modules['prox_l_l'](q_l_k)
        L_n_k = modules['prox_l_n'](q_n_k)

        return {
            'R': R_k,
            'L_l': L_l_k,
            'L_n': L_n_k
        }

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, 'is_pass_conv') and m.is_pass_conv:
                    continue
                if hasattr(m, 'is_last_conv') and m.is_last_conv:
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    nn.init.kaiming_normal_(
                        m.weight,
                        a=0.01,
                        mode='fan_out',
                        nonlinearity='leaky_relu'
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)


class RetinexUnfoldingNetUnsupervised_Test(nn.Module):
    def __init__(self, num_steps: int = 3, init_lambda=0.2, gamma=0.8):
        super().__init__()
        self.num_steps = num_steps
        self.gamma = gamma

        self.decomposer = InitialDecomposer()

        self.gradient_steps = nn.ModuleList([
            nn.ModuleDict({
                'p': P_Update(init_lambda),
                'q_low': Q_Low_Update(init_lambda),
                'q_norm': Q_Norm_Update(init_lambda),
                'prox_r': P_ProxNet(),
                'prox_l_l': Q_Low_ProxNet(),
                'prox_l_n': Q_Low_ProxNet(),
            }) for i in range(num_steps)
        ])

    def forward(self, I_l):
        I_n = torch.pow(I_l, self.gamma)

        R_l, L_l = self.decomposer(I_l)
        R_n, L_n = self.decomposer(I_n)

        R_k, L_l_k, L_n_k, = R_l, L_l, L_n

        for step in range(self.num_steps):
            modules = self.gradient_steps[step]
            q_k = modules['p'](R_k, L_l_k, I_l, L_n_k, I_n)
            R_k = modules['prox_r'](q_k)
            q_l_k = modules['q_low'](L_l_k, R_k, I_l)
            q_n_k = modules['q_norm'](L_n_k, R_k, I_n)
            L_l_k = modules['prox_l_l'](q_l_k)
            L_n_k = modules['prox_l_n'](q_n_k)

        return R_k, L_l_k, L_n_k
