from utils.sharpness import hv_computation_dataloader, group_product, get_params_grad, normalization, group_add, orthnormal
import math, torch, os
from args.args_utils import *
import numpy as np
from tqdm import tqdm
from numpy.polynomial.chebyshev import chebval

def rescale_H_action(Hv, v, lambda_max, lambda_min):
    return (2 * Hv - (lambda_max + lambda_min) * v) / (lambda_max - lambda_min)

def jackson_kernel(M):
    """返回 Jackson kernel 权重 g_k^{(M)}，用于 Chebyshev 系数平滑"""
    g = []
    for k in range(M + 1):
        coeff = ((M - k + 1) * np.cos(np.pi * k / (M + 1)) +
                 np.sin(np.pi * k / (M + 1)) / np.tan(np.pi / (M + 1)))
        g_k = coeff / (M + 1)
        g.append(g_k)
    return g

def compute_miu(model, criterion, dataloader, param_names, M, N, lambda_max, lambda_min):
    params, _ = get_params_grad(model, param_names)
    parameters = params
    n = torch.norm(parameters[0], p=1) # H is n times n

    zeta_list = [0 for _ in range(M+1)]
    for l in tqdm(range(N)):
        v = v = [torch.randint_like(p, high=2).cuda(parser_args.gpu) * 2 - 1 for p in parameters
            ] # same as params
        v_list = [v]
        for k in range(M+1):
            zeta_list[k] += group_product(v_list[0],v_list[k])
            if k==0:
                _, Hv = hv_computation_dataloader(parameters, dataloader, model, criterion, v_list[k], param_names)
                Hv = rescale_H_action(Hv[0], v_list[k][0], lambda_max, lambda_min)
                v_list.append([Hv])
            else:
                _, Hv = hv_computation_dataloader(parameters, dataloader, model, criterion, v_list[k], param_names)
                Hv = rescale_H_action(Hv[0], v_list[k][0], lambda_max, lambda_min)
                v_list.append([2*Hv - v_list[k-1][0]])
    
    nomalize_zeta_list = [i/N for i in zeta_list]
    miu_list = [2/(n*math.pi)*i for i in nomalize_zeta_list]
    miu_list[0] = miu_list[0]/2

    jackson = jackson_kernel(M)
    miu_list = [mu * g for mu, g in zip(miu_list, jackson)]
    miu_list = [i.item() for i in miu_list]

    return miu_list

def density_f(miu_list, t):
    # φ̃_M(t) = 1 / sqrt(1 - t^2) * sum_{k=0}^{M} μ_k T_k(t)
    t = np.asarray(t)
    if np.any(np.abs(t) >= 1):
        raise ValueError("t 必须在 (-1, 1) 区间内")

    numerator = chebval(t, miu_list)  # 计算 ∑ μ_k T_k(t)
    denominator = np.sqrt(1 - t**2)
    return numerator / denominator

def flatten_tensor_list(tensor_list):
    return torch.cat([t.reshape(-1) for t in tensor_list])

def density(batches, model, loss_f, param_names, iter=100, n_v=1, a=1, min_iter=True, dir=''):
    import time
    """
    compute T using stochastic lanczos algorithm (SLQ)
    iter: number of iterations used to compute T
    n_v: number of SLQ runs
    """

    model.zero_grad()
    params, grads = get_params_grad(model, param_names)
    parameters = params

    vhav_list = []
    for k in tqdm(range(n_v)):
        v = [
            torch.randint_like(p, high=2).cuda(parser_args.gpu)
            for p in parameters
        ]
        # generate Rademacher random variables
        for v_i in v:
            v_i[v_i == 0] = -1
        v_flat = flatten_tensor_list(v)
        v = normalization(v)

        # standard lanczos algorithm initlization
        v_list = [v]
        w_list = []
        alpha_list = []
        beta_list = []
        ############### Lanczos
        for i in range(iter):
            #print(i, iter)
            if i == 0:
                one_time = time.time()
                two_time = time.time()
            else:
                one_time = two_time
                two_time = time.time()
                diff_time = two_time - one_time
                need_time = (iter-i)*diff_time/(3600)
                #print("need time {} hours".format(need_time))
            model.zero_grad()
            w_prime = [torch.zeros(p.size()).cuda(parser_args.gpu) for p in parameters]
            if i == 0:
                _, w_prime = hv_computation_dataloader(parameters, batches, model, loss_f, v, param_names)
                alpha = group_product(w_prime, v)
                alpha_list.append(alpha.cpu().item())
                w = group_add(w_prime, v, alpha=-alpha)
                w_list.append(w)
            else:
                beta = torch.sqrt(group_product(w, w))
                beta_list.append(beta.cpu().item())
                if beta_list[-1] != 0.:
                    # We should re-orth it
                    v = orthnormal(w, v_list)
                    v_list.append(v)
                else:
                    # generate a new vector
                    w = [torch.randn(p.size()).cuda(parser_args.gpu) for p in parameters]
                    v = orthnormal(w, v_list)
                    v_list.append(v)
                _, w_prime = hv_computation_dataloader(parameters, batches, model, loss_f, v, param_names)
                alpha = group_product(w_prime, v)
                alpha_list.append(alpha.cpu().item())
                w_tmp = group_add(w_prime, v, alpha=-alpha)
                w = group_add(w_tmp, v_list[-2], alpha=-beta)

        T = torch.zeros(iter, iter).cuda(parser_args.gpu)
        for i in range(len(alpha_list)):
            T[i, i] = alpha_list[i]
            if i < len(alpha_list) - 1:
                T[i + 1, i] = beta_list[i]
                T[i, i + 1] = beta_list[i]

        eigvals, eigvecs = torch.linalg.eigh(T)  # T is real symmetric
        T_a = eigvecs @ torch.diag(eigvals**a) @ eigvecs.T  # f(T) = T^a

        v_norm = torch.norm(v_flat)
        e1 = torch.zeros(iter, dtype=T_a.dtype, device=T_a.device)
        e1[0] = 1.0
        scalar_estimate = v_norm**2 * torch.dot(e1, T_a @ e1)
        vhav_list.append(scalar_estimate.item())

        power_dict = torch.load(dir+'.pth', weights_only=False)
        power_dict['T'].append(T)
        power_dict['v_norm'].append(v_norm)
    
        torch.save(power_dict, dir+'.pth')

    return np.mean(vhav_list)

def check_density(batches, model, loss_f, param_names, iter=100, n_v=1, a=1, min_iter=True, dir=''):
    flag = True
    if os.path.exists(dir+'.pth'):
        power_dict = torch.load(dir+'.pth', weights_only=False)
        if len(power_dict['T']) > n_v-1:
            flag = False
    else:
        power_dict = {
            "T": [],
            "v_norm": []
        }
        torch.save(power_dict, dir+'.pth')

    if flag:
        density(batches, model, loss_f, param_names, iter=iter, n_v=n_v, a=a, min_iter=min_iter, dir=dir)


def compute_T(batches, model, loss_f, param_names, iter=100, n_v=1, dir=''):
    import time
    """
    compute estimated eigenvalue density using stochastic lanczos algorithm (SLQ)
    iter: number of iterations used to compute trace
    n_v: number of SLQ runs
    """

    model.zero_grad()
    params, grads = get_params_grad(model, param_names)
    parameters = params

    T_dict = {
        "T": []
    }
    for k in tqdm(range(n_v)):
        v = [
            torch.randint_like(p, high=2).cuda(parser_args.gpu)
            for p in parameters
        ]
        # generate Rademacher random variables
        for v_i in v:
            v_i[v_i == 0] = -1
        v_flat = flatten_tensor_list(v)
        v = normalization(v)

        # standard lanczos algorithm initlization
        v_list = [v]
        w_list = []
        alpha_list = []
        beta_list = []
        ############### Lanczos
        for i in range(iter):
            #print(i, iter)
            if i == 0:
                one_time = time.time()
                two_time = time.time()
            else:
                one_time = two_time
                two_time = time.time()
                diff_time = two_time - one_time
                need_time = (iter-i)*diff_time/(3600)
                #print("need time {} hours".format(need_time))
            model.zero_grad()
            w_prime = [torch.zeros(p.size()).cuda(parser_args.gpu) for p in parameters]
            if i == 0:
                _, w_prime = hv_computation_dataloader(parameters, batches, model, loss_f, v, param_names)
                alpha = group_product(w_prime, v)
                alpha_list.append(alpha.cpu().item())
                w = group_add(w_prime, v, alpha=-alpha)
                w_list.append(w)
            else:
                beta = torch.sqrt(group_product(w, w))
                beta_list.append(beta.cpu().item())
                if beta_list[-1] != 0.:
                    # We should re-orth it
                    v = orthnormal(w, v_list)
                    v_list.append(v)
                else:
                    # generate a new vector
                    w = [torch.randn(p.size()).cuda(parser_args.gpu) for p in parameters]
                    v = orthnormal(w, v_list)
                    v_list.append(v)
                _, w_prime = hv_computation_dataloader(parameters, batches, model, loss_f, v, param_names)
                alpha = group_product(w_prime, v)
                alpha_list.append(alpha.cpu().item())
                w_tmp = group_add(w_prime, v, alpha=-alpha)
                w = group_add(w_tmp, v_list[-2], alpha=-beta)

        T = torch.zeros(iter, iter).cuda(parser_args.gpu)
        for i in range(len(alpha_list)):
            T[i, i] = alpha_list[i]
            if i < len(alpha_list) - 1:
                T[i + 1, i] = beta_list[i]
                T[i, i + 1] = beta_list[i]

        # eigvals, eigvecs = torch.linalg.eigh(T)  # T is real symmetric

        T_dict['T'].append(T)
        torch.save(T_dict, dir+'.pth')

def compute_var_for_plot(T_list):
    eigen_list_full, weight_list_full = [], []
    for T in T_list:
        a1_, b1_ = torch.linalg.eigh(T)
        eigen_list = a1_
        weight_list = b1_**2
        eigen_list_full.append(list(eigen_list.cpu().numpy()))
        weight_list_full.append(list(weight_list.cpu().numpy()))

    return eigen_list_full, weight_list_full