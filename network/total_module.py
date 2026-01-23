import torch

from torch import nn

from network.math_module import P_Update, Q_Low_Update, Q_Norm_Update
from network.illumination_low_prox import Q_Low_ProxNet
from network.init_decom import InitialDecomposer
from network.reflection_prox import P_ProxNet


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
