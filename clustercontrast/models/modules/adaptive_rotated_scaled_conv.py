import torch
import torch.nn as nn
from torch.nn import functional as F


__all__ = ['AdaptiveRotatedScaledConv2d']


def _get_rotation_matrix(thetas, scales): # scales should be of the same shape as thetas
    bs, g = thetas.shape # bs = batch_size, g = num_experts
    device = thetas.device
    thetas = thetas.reshape(-1)  # [bs, g] --> [bs x g]
    scales = scales.reshape(-1)  # [bs, g] --> [bs x g]

    x = torch.mul(torch.cos(thetas), scales)
    y = torch.mul(torch.sin(thetas), scales)
    y_prime = -torch.mul(torch.sin(thetas), scales) # for negative rotation
    x = x.unsqueeze(0).unsqueeze(0)  # shape = [1, 1, bs * g]
    y = y.unsqueeze(0).unsqueeze(0)
    y_prime = y_prime.unsqueeze(0).unsqueeze(0)
    a = x - y  # shape = [1, 1, bs * g]
    b = x * y  # shape = [1, 1, bs * g]
    c = x + y
    d = a * c
    e = a + c
    a_prime = x - y_prime
    b_prime = x * y_prime
    c_prime = x + y_prime
    d_prime = a_prime * c_prime
    e_prime = a_prime + c_prime

    rot_mat_positive_big_1 = torch.cat((
        torch.cat((a, 1-a, torch.zeros(1, 7, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs*g, device=device), 1-y, y, torch.zeros(1, 6, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 2, bs*g, device=device), a, torch.zeros(1, 2, bs*g, device=device), 1-a, torch.zeros(1, 3, bs*g, device=device)), dim=1),
        torch.cat((y,torch.zeros(1, 2, bs*g, device=device),1-y,torch.zeros(1, 5, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), torch.ones(1, 1, bs*g, device=device), torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 5, bs*g, device=device),1-y,torch.zeros(1, 2, bs*g, device=device),y), dim=1),
        torch.cat((torch.zeros(1, 3, bs*g, device=device),1-a,torch.zeros(1, 2, bs*g, device=device),a,torch.zeros(1, 2, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 6, bs*g, device=device),y,1-y,torch.zeros(1, 1, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 7, bs*g, device=device), 1-a, a), dim=1)
    ), dim=0)  # shape = [k^2, k^2, bs*g]

    rot_mat_positive_big_2 = torch.cat((
        torch.cat((a, 1-a, torch.zeros(1, 7, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs*g, device=device), x-b, b, torch.zeros(1, 1, bs*g, device=device), 1-c+b, y-b, torch.zeros(1, 3, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 2, bs*g, device=device), a, torch.zeros(1, 2, bs*g, device=device), 1-a, torch.zeros(1, 3, bs*g, device=device)), dim=1),
        torch.cat((b, y-b, torch.zeros(1,1 , bs*g, device=device), x-b, 1-c+b, torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), torch.ones(1, 1, bs*g, device=device), torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), 1-c+b, x-b, torch.zeros(1, 1, bs*g, device=device), y-b, b), dim=1),
        torch.cat((torch.zeros(1, 3, bs*g, device=device), 1-a, torch.zeros(1, 2, bs*g, device=device), a, torch.zeros(1, 2, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 3, bs*g, device=device), y-b, 1-c+b, torch.zeros(1, 1, bs*g, device=device), b, x-b, torch.zeros(1, 1, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 7, bs*g, device=device), 1-a, a), dim=1)
    ), dim=0)  # shape = [k^2, k^2, bs*g]

    rot_mat_positive_small_1 = torch.cat((
        torch.cat((d, a-d, torch.zeros(1, 1, bs*g, device=device),c-d, 1-e+d, torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs*g, device=device),x-b, b, torch.zeros(1, 1, bs*g, device=device),1-c+b, y-b, torch.zeros(1, 3, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs*g, device=device),c-d, d, torch.zeros(1, 1, bs*g, device=device), 1-e+d, a-d, torch.zeros(1, 3, bs*g, device=device)), dim=1),
        torch.cat((b,y-b,torch.zeros(1, 1, bs*g, device=device),x-b, 1-c+b,torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), torch.ones(1, 1, bs*g, device=device), torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), 1-c+b, x-b, torch.zeros(1, 1, bs*g, device=device), y-b, b), dim=1),
        torch.cat((torch.zeros(1, 3, bs*g, device=device), a-d, 1-e+d, torch.zeros(1, 1, bs*g, device=device), d, c-d, torch.zeros(1, 1, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 3, bs*g, device=device), y-b, 1-c+b, torch.zeros(1, 1, bs*g, device=device), b, x-b, torch.zeros(1, 1, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), 1-e+d, c-d, torch.zeros(1, 1, bs*g, device=device), a-d, d), dim=1)
    ), dim=0)  # shape = [k^2, k^2, bs*g]

    rot_mat_positive_small_2 = torch.cat((
        torch.cat((a, 1-a, torch.zeros(1, 7, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs*g, device=device), x-b, b, torch.zeros(1, 1, bs*g, device=device), 1-c+b, y-b, torch.zeros(1, 3, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 2, bs*g, device=device), a, torch.zeros(1, 2, bs*g, device=device), 1-a, torch.zeros(1, 3, bs*g, device=device)), dim=1),
        torch.cat((b, y-b, torch.zeros(1,1 , bs*g, device=device), x-b, 1-c+b, torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), torch.ones(1, 1, bs*g, device=device), torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), 1-c+b, x-b, torch.zeros(1, 1, bs*g, device=device), y-b, b), dim=1),
        torch.cat((torch.zeros(1, 3, bs*g, device=device), 1-a, torch.zeros(1, 2, bs*g, device=device), a, torch.zeros(1, 2, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 3, bs*g, device=device), y-b, 1-c+b, torch.zeros(1, 1, bs*g, device=device), b, x-b, torch.zeros(1, 1, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 7, bs*g, device=device), 1-a, a), dim=1)
    ), dim=0)  # shape = [k^2, k^2, bs*g]

    rot_mat_negative_big_1 = torch.cat((
        torch.cat((c_prime, torch.zeros(1, 2, bs*g, device=device), 1-c_prime, torch.zeros(1, 5, bs*g, device=device)), dim=1),
        torch.cat((y_prime, 1-y_prime, torch.zeros(1, 7, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs*g, device=device), 1-c_prime, c_prime, torch.zeros(1, 6, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 3, bs*g, device=device), 1-y_prime, torch.zeros(1, 2, bs*g, device=device), y_prime, torch.zeros(1, 2, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), torch.ones(1, 1, bs*g, device=device), torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 2, bs*g, device=device), y_prime, torch.zeros(1, 2, bs*g, device=device), 1-y_prime, torch.zeros(1, 3, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 6, bs*g, device=device),c_prime, 1-c_prime, torch.zeros(1, 1, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 7, bs*g, device=device),1-y_prime, y_prime), dim=1),
        torch.cat((torch.zeros(1, 5, bs*g, device=device),1-c_prime, torch.zeros(1, 2, bs*g, device=device), c_prime), dim=1)
    ), dim=0)  # shape = [k^2, k^2, bs*g]

    rot_mat_negative_big_2 = torch.cat((
        torch.cat((c_prime, torch.zeros(1, 2, bs*g, device=device), 1-c_prime, torch.zeros(1, 5, bs*g, device=device)), dim=1),
        torch.cat((b_prime, x - b_prime, torch.zeros(1, 1, bs*g, device=device), y_prime - b_prime, 1 - c_prime + b_prime, torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs*g, device=device), 1-c_prime, c_prime, torch.zeros(1, 6, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 3, bs*g, device=device), x-b_prime, 1-c_prime+b_prime, torch.zeros(1, 1, bs*g, device=device), b_prime, y_prime - b_prime, torch.zeros(1, 1, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), torch.ones(1, 1, bs*g, device=device), torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs*g, device=device), y_prime - b_prime, b_prime, torch.zeros(1, 1, bs*g, device=device), 1- c_prime + b_prime, x - b_prime, torch.zeros(1, 3, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 6, bs*g, device=device), c_prime, 1-c_prime, torch.zeros(1, 1, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), 1- c_prime + b_prime, y_prime - b_prime, torch.zeros(1, 1, bs*g, device=device), x-b_prime, b_prime), dim=1),
        torch.cat((torch.zeros(1, 5, bs*g, device=device), 1-c_prime, torch.zeros(1, 2, bs*g, device=device), c_prime), dim=1)
    ), dim=0)  # shape = [k^2, k^2, bs*g]

    rot_mat_negative_small_1 = torch.cat((
        torch.cat((d_prime, c_prime - d_prime, torch.zeros(1, 1, bs*g, device=device), a_prime - d_prime , 1 - e_prime + d_prime, torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((b_prime, x - b_prime, torch.zeros(1, 1, bs*g, device=device), y_prime - b_prime, 1 - c_prime + b_prime, torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs*g, device=device), a_prime - d_prime, d_prime, torch.zeros(1, 1, bs*g, device=device), 1 - e_prime + d_prime, c_prime - d_prime, torch.zeros(1, 3, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs*g, device=device), y_prime - b_prime, b_prime, torch.zeros(1, 1, bs*g, device=device),1 - c_prime + b_prime, x - b_prime, torch.zeros(1, 3, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), torch.ones(1, 1, bs*g, device=device), torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 3, bs*g, device=device), x - b_prime, 1 - c_prime + b_prime, torch.zeros(1, 1, bs*g, device=device), b_prime, y_prime - b_prime, torch.zeros(1, 1, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 3, bs*g, device=device), c_prime - d_prime, 1 - e_prime + d_prime, torch.zeros(1, 1, bs*g, device=device), d_prime, a_prime - d_prime,torch.zeros(1, 1, bs*g, device=device) ), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), 1 - c_prime + b_prime, y_prime - b_prime, torch.zeros(1, 1, bs*g, device=device), x - b_prime, b_prime), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), 1 - e_prime + d_prime, a_prime - d_prime, torch.zeros(1, 1, bs*g, device=device), c_prime - d_prime, d_prime), dim=1)
    ), dim=0)  # shape = [k^2, k^2, bs*g]

    rot_mat_negative_small_2 = torch.cat((
        torch.cat((c_prime, torch.zeros(1, 2, bs*g, device=device), 1-c_prime, torch.zeros(1, 5, bs*g, device=device)), dim=1),
        torch.cat((b_prime, x - b_prime, torch.zeros(1, 1, bs*g, device=device), y_prime - b_prime, 1 - c_prime + b_prime, torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs*g, device=device), 1-c_prime, c_prime, torch.zeros(1, 6, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 3, bs*g, device=device), x-b_prime, 1-c_prime+b_prime, torch.zeros(1, 1, bs*g, device=device), b_prime, y_prime - b_prime, torch.zeros(1, 1, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), torch.ones(1, 1, bs*g, device=device), torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs*g, device=device), y_prime - b_prime, b_prime, torch.zeros(1, 1, bs*g, device=device), 1- c_prime + b_prime, x - b_prime, torch.zeros(1, 3, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 6, bs*g, device=device), c_prime, 1-c_prime, torch.zeros(1, 1, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), 1- c_prime + b_prime, y_prime - b_prime, torch.zeros(1, 1, bs*g, device=device), x-b_prime, b_prime), dim=1),
        torch.cat((torch.zeros(1, 5, bs*g, device=device), 1-c_prime, torch.zeros(1, 2, bs*g, device=device), c_prime), dim=1)
    ), dim=0)  # shape = [k^2, k^2, bs*g]

    mask_thetas_positive = (thetas >= 0).unsqueeze(0).unsqueeze(0).float()  # shape = [1, 1, bs*g]
    mask_thetas_negative = (thetas < 0).unsqueeze(0).unsqueeze(0).float()
    mask_scales_big = (scales >= 1).unsqueeze(0).unsqueeze(0).float()  # shape = [1, 1, bs*g]
    mask_scales_small = (scales < 1).unsqueeze(0).unsqueeze(0).float()

    mask_positive_big = mask_thetas_positive * mask_scales_big
    mask_positive_small = mask_thetas_positive * mask_scales_small
    mask_negative_big = mask_thetas_negative * mask_scales_big
    mask_negative_small = mask_thetas_negative * mask_scales_small

    thetas_positive_big = mask_positive_big * (thetas.unsqueeze(0).unsqueeze(0))        # thetas [bs*n] adjust to [1, 1, bs*g]
    thetas_positive_small = mask_positive_small * (thetas.unsqueeze(0).unsqueeze(0))
    thetas_negative_big = mask_negative_big * (thetas.unsqueeze(0).unsqueeze(0))
    thetas_negative_small = mask_negative_small * (thetas.unsqueeze(0).unsqueeze(0))

    # thresholds of scales and thetas
    thres_positive_big = 1 / (torch.cos(thetas_positive_big)) # should be angel, not bool value!! # [1, 1, bs*g]
    thres_positive_small = 1/(torch.cos(thetas_positive_small) + torch.sin(thetas_positive_small))
    thres_negative_big = 1/(torch.cos(-thetas_negative_big))
    thres_negative_small = 1/(torch.cos(thetas_negative_small) - torch.sin(thetas_negative_small))

    # 8 mask senarios
    mask_positive_big_1 = (thetas_positive_big >= thres_positive_big) * mask_positive_big
    mask_positive_big_2 = (thetas_positive_big < thres_positive_big) * mask_positive_big
    mask_positive_small_1 = (thetas_positive_small < thres_positive_small) * mask_positive_small
    mask_positive_small_2 = (thetas_positive_small >= thres_positive_small) * mask_positive_small
    mask_negative_big_1 = (thetas_negative_big >= thres_negative_big) * mask_negative_big
    mask_negative_big_2 = (thetas_negative_big < thres_negative_big) * mask_negative_big
    mask_negative_small_1 = (thetas_negative_small < thres_negative_small) * mask_negative_small
    mask_negative_small_2 = (thetas_negative_small >= thres_negative_small) * mask_negative_small

    rot_mat = (mask_positive_big_1*rot_mat_positive_big_1 + mask_positive_big_2*rot_mat_positive_big_2\
               + mask_positive_small_1*rot_mat_positive_small_1 + mask_positive_small_2*rot_mat_positive_small_2\
               + mask_negative_big_1*rot_mat_negative_big_1+ mask_negative_big_2*rot_mat_negative_big_2 \
               + mask_negative_small_1*rot_mat_negative_small_1 + mask_negative_small_2*rot_mat_negative_small_2)  # shape = [k*k, k*k, bs*g]
    rot_mat = rot_mat.permute(2, 0, 1)                                    # shape = [bs*g, k*k, k*k]
    rot_mat = rot_mat.reshape(bs, g, rot_mat.shape[1], rot_mat.shape[2])  # shape = [bs, g, k*k, k*k]
    return rot_mat

def batch_rotate_multiweight(weights, lambdas, thetas, scales):
    """
    Let
        batch_size = b
        kernel_number = n
        kernel_size = 3
    Args:
        weights: tensor, shape = [kernel_number, Cout, Cin, k, k]
        thetas: tensor of thetas,  shape = [batch_size, kernel_number]
        scales: tensor of scales,  shape = [batch_size, kernel_number]
    Return:
        weights_out: tensor, shape = [batch_size x Cout, Cin // groups, k, k]
    """
    assert(thetas.shape == lambdas.shape)
    assert(scales.shape == lambdas.shape)
    assert(lambdas.shape[1] == weights.shape[0])

    b = thetas.shape[0]
    n = thetas.shape[1]
    k = weights.shape[-1]
    _, Cout, Cin, _, _ = weights.shape 

    # Stage 1:
    # input: thetas: [b, n]
    #        lambdas: [b, n]
    # output: rotation_matrix: [b, n, 9, 9] (with gate) --> [b*9, n*9]

    #       Sub_Stage 1.1:
    #       input: [b, n] kernel
    #       output: [b, n, 9, 9] rotation matrix
    rotation_matrix = _get_rotation_matrix(thetas, scales)

    #       Sub_Stage 1.2:
    #       input: [b, n, 9, 9] rotation matrix
    #              [b, n] lambdas
    #          --> [b, n, 1, 1] lambdas
    #          --> [b, n, 1, 1] lambdas dot [b, n, 9, 9] rotation matrix
    #          --> [b, n, 9, 9] rotation matrix with gate (done)
    #       output: [b, n, 9, 9] rotation matrix with gate
    lambdas = lambdas.unsqueeze(2).unsqueeze(3)
    rotation_matrix = torch.mul(rotation_matrix, lambdas)

    #       Sub_Stage 1.3: Reshape
    #       input: [b, n, 9, 9] rotation matrix with gate
    #       output: [b*9, n*9] rotation matrix with gate
    rotation_matrix = rotation_matrix.permute(0, 2, 1, 3)
    rotation_matrix = rotation_matrix.reshape(b*9, n*9)

    # Stage 2: Reshape
    # input: weights: [n, Cout, Cin, 3, 3]
    #             --> [n, 3, 3, Cout, Cin]
    #             --> [n*9, Cout*Cin] done
    # output: weights: [n*9, Cout*Cin]
    weights = weights.permute(0, 3, 4, 1, 2)
    weights = weights.contiguous().view(n*9, Cout*Cin)

    # Stage 3: torch.mm
    # [b*9, n*9] x [n*9, Cout*Cin]
    # --> [b*9, Cout*Cin]
    weights = torch.mm(rotation_matrix, weights)

    # Stage 4: Reshape Back
    # input: [b*9, Cout*Cin]
    #    --> [b, 3, 3, Cout, Cin]
    #    --> [b, Cout, Cin, 3, 3]
    #    --> [b * Cout, Cin, 3, 3] done
    # output: [b * Cout, Cin, 3, 3]
    weights = weights.contiguous().view(b, 3, 3, Cout, Cin)
    weights = weights.permute(0, 3, 4, 1, 2)
    weights = weights.reshape(b * Cout, Cin, 3, 3)

    return weights


class AdaptiveRotatedScaledConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1, bias=False,
                 kernel_number=1, rounting_func=None, rotate_func=batch_rotate_multiweight):
        super().__init__()
        self.kernel_number = kernel_number
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.rounting_func = rounting_func
        self.rotate_func = rotate_func

        self.weight = nn.Parameter(
            torch.Tensor(
                kernel_number, 
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
            )
        )
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # get alphas, angles
        # # [bs, Cin, h, w] --> [bs, n_theta], [bs, n_theta]
        alphas, angles, scales = self.rounting_func(x)

        # rotate weight
        # # [Cout, Cin, k, k] --> [bs * Cout, Cin, k, k]
        rotated_weight = self.rotate_func(self.weight, alphas, angles, scales)

        # reshape images
        bs, Cin, h, w = x.shape
        x = x.reshape(1, bs * Cin, h, w)  # [1, bs * Cin, h, w]
        
        # adaptive conv over images using group conv
        out = F.conv2d(input=x, weight=rotated_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=(self.groups * bs))
        
        # reshape back
        out = out.reshape(bs, self.out_channels, *out.shape[2:])
        return out
        

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_number={kernel_number}'
             ', kernel_size={kernel_size}, stride={stride}, bias={bias}')
             
        if self.padding != (0,) * len([self.padding]):
            s += ', padding={padding}'
        if self.dilation != (1,) * len([self.dilation]):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        return s.format(**self.__dict__)
    