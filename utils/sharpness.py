import torch
import utils
import copy
import math
import tqdm
import numpy as np
from args.args_utils import *
from utils.utils import *

def get_params_grad_wo_name(model):
    params = []
    grads = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1:
            pass
        else:
            params.append(param)
            grads.append(0. if param.grad is None else param.grad + 0.)
    return params, grads

def get_params_grad_with_name(model, param_names):
    params = []
    grads = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1:
            pass
        else:
            if name == param_names:
                params.append(param)
                grads.append(0. if param.grad is None else param.grad + 0.)

    return params, grads

def get_params_grad(model, param_names):
    if param_names is not None:
        #print(f"get param weights with name: {param_names}")
        params, grads = get_params_grad_with_name(model, param_names)
    else:
        #print("get all trainable weights")
        params, grads = get_params_grad_wo_name(model)
    
    return params, grads

def group_product(xs, ys):
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])

def group_add(params, update, alpha=1):
    for i, p in enumerate(params):
        params[i].data.add_(update[i] * alpha)
    return params

def normalization(v):
    s = group_product(v, v)
    s = s**0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    return v

def orthnormal(w, v_list):
    for v in v_list:
        w = group_add(w, v, alpha=-group_product(w, v))
    return normalization(w)

def hv_computation_dataloader(parameters, batches, model, loss_f, v, param_names): #='model.to_patch_embedding.1.weight'    
    num_data = 0.
    THv = [torch.zeros(p.size()).cuda(parser_args.gpu) for p in parameters
            ]  # accumulate result
    for i_batch, (inputs, targets) in enumerate(batches):
        inputs, targets = inputs.cuda(parser_args.gpu), targets.cuda(parser_args.gpu)
        model.zero_grad()
        outputs = model(inputs)
        loss = loss_f(outputs, targets)
        grads = torch.autograd.grad(loss, parameters, create_graph=True)
        # grads = torch.autograd.grad(loss, parameters, create_graph=True, allow_unused=True) # allow_unused=True is set for imagenet finetuning from clip/vit-b/32
        #loss.backward(create_graph=True)
        #params, grads = get_params_grad(model, param_names)
        model.zero_grad()
        
        Hv = torch.autograd.grad(grads, parameters, grad_outputs=v, only_inputs=True, retain_graph=False)
        THv = [
                THv1 + Hv1 * float(inputs.size(0)) + 0.
                for THv1, Hv1 in zip(THv, Hv)
            ]
        num_data += float(inputs.size(0))

        # del loss, grads, Hv, outputs
        # torch.cuda.empty_cache()
        
    THv = [THv1 / float(num_data) for THv1 in THv]
    vHv = group_product(THv, v).cpu().item()

    return vHv, THv

def cumpute_eigenvalues(batches, model, loss_f, param_names, maxIter=100, tol=1e-3, top_n=1):
    model.zero_grad()
    params, grads = get_params_grad(model, param_names)
    parameters = params
    
    eigenvalues = []
    eigenvectors = []

    computed_dim = 0
    while computed_dim < top_n:
        eigenvalue = None
        v = [torch.randn(p.size()).cuda(parser_args.gpu) for p in parameters
            ]
        v = normalization(v)
        for i in range(maxIter):
            v = orthnormal(v, eigenvectors)
            model.zero_grad()

            tmp_eigenvalue, Hv = hv_computation_dataloader(parameters, batches, model, loss_f, v, param_names)
            v = normalization(Hv)

            if eigenvalue == None:
                eigenvalue = tmp_eigenvalue
            else:
                if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) +1e-6) < tol:
                    break
                else:
                    eigenvalue = tmp_eigenvalue
                print(eigenvalue, computed_dim, i)
        eigenvalues.append(eigenvalue)
        eigenvectors.append(v)
        computed_dim += 1

    return eigenvalues, eigenvectors

def power_iteration(batches, model, loss_f, param_names, maxIter=100, tol=1e-3):
    model.zero_grad()
    params, grads = get_params_grad(model, param_names)
    parameters = params

    b_k = [torch.randn(p.size()).cuda(parser_args.gpu) for p in parameters
            ]
    for i in range(maxIter):
        _, b_k1 = hv_computation_dataloader(parameters, batches, model, loss_f, b_k, param_names)
        b_k = normalization(b_k1)

    _, Ab_k = hv_computation_dataloader(parameters, batches, model, loss_f, b_k, param_names)
    max_eigen = group_product(Ab_k, b_k)/group_product(b_k, b_k)

    return max_eigen.item()

def power_iteration_with_index(batches, model, loss_f, param_names, maxIter=100, tol=1e-3, index=1):
    model.zero_grad()
    params, grads = get_params_grad(model, param_names)
    parameters = params

    b_k_ori = [torch.randn(p.size()).cuda(parser_args.gpu) for p in parameters
            ]
    b_k = [torch.zeros(p.size()).cuda(parser_args.gpu) for p in parameters
            ]
    b_k[0][1,index] = b_k_ori[0][1,index]
    for i in range(maxIter):
        _, b_k1 = hv_computation_dataloader(parameters, batches, model, loss_f, b_k, param_names)
        b_k = normalization(b_k1)

    _, Ab_k = hv_computation_dataloader(parameters, batches, model, loss_f, b_k, param_names)
    max_eigen = group_product(Ab_k, b_k)/group_product(b_k, b_k)

    return max_eigen.item()

import torch

def lanczos_min_eigenvalue(model, param_names, batches, loss_f, max_iter=100, dtype=torch.float32, device=f'cuda:{parser_args.gpu}'):
    model.zero_grad()
    params, grads = get_params_grad(model, param_names)
    parameters = params

    # 初始化第一个向量 v_0
    v = [
            torch.randint_like(p, high=2).cuda(parser_args.gpu)
            for p in parameters
        ]
    v = v[0] / torch.norm(v[0])

    Vs = [v]  # 保存所有 v_j
    alphas = []
    betas = []

    beta = 0
    v_prev = torch.zeros_like(v)

    for i in range(max_iter):
        _,w = hv_computation_dataloader(parameters, batches, model, loss_f, [v], param_names)
        w = w[0]
        alpha = group_product([v], [w])
        w = w - alpha * v - beta * v_prev

        beta = torch.norm(w).item()
        if beta < 1e-10:
            break

        v_prev = v
        v = w / beta
        Vs.append(v)

        alphas.append(alpha)
        betas.append(beta)

    # 构造对称三对角矩阵 T
    T = torch.diag(torch.tensor(alphas, dtype=dtype, device=device))
    if len(betas) > 1:
        off_diag = torch.tensor(betas[:-1], dtype=dtype, device=device)
        T += torch.diag(off_diag, diagonal=1) + torch.diag(off_diag, diagonal=-1)

    # 求 T 的最小特征值
    eigvals = torch.linalg.eigvalsh(T.cpu())
    return eigvals[0].item(), eigvals[-1].item()

def density(batches, model, loss_f, param_names, iter=100, n_v=1, min_iter=True):
    import time
    """
    compute estimated eigenvalue density using stochastic lanczos algorithm (SLQ)
    iter: number of iterations used to compute trace
    n_v: number of SLQ runs
    """

    model.zero_grad()
    params, grads = get_params_grad(model, param_names)
    parameters = params

    eigen_list_full = []
    weight_list_full = []

    for k in range(n_v):
        v = [
            torch.randint_like(p, high=2).cuda(parser_args.gpu)
            for p in parameters
        ]
        # generate Rademacher random variables
        for v_i in v:
            v_i[v_i == 0] = -1
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

            if min_iter:
                T1 = torch.zeros(i+1, i+1).cuda(parser_args.gpu)
                for i in range(len(alpha_list)):
                    T1[i, i] = alpha_list[i]
                    if i < len(alpha_list) - 1:
                        T1[i + 1, i] = beta_list[i]
                        T1[i, i + 1] = beta_list[i]
                a1_, b1_ = torch.linalg.eig(T1)

                sorted_list1 = sorted(a1_, key=abs, reverse=True)
                if i == 0:
                    last_min_ev =torch.tensor(0.)
                min_ev = sorted_list1[-1]
                diff_ev = abs(abs(min_ev)-abs(last_min_ev))
                last_min_ev = min_ev
                print(sorted_list1[0], last_min_ev)

        T = torch.zeros(iter, iter).cuda(parser_args.gpu)
        for i in range(len(alpha_list)):
            T[i, i] = alpha_list[i]
            if i < len(alpha_list) - 1:
                T[i + 1, i] = beta_list[i]
                T[i, i + 1] = beta_list[i]
        a1_, b1_ = torch.linalg.eig(T)

        eigen_list = a1_.real
        weight_list = b1_**2
        eigen_list_full.append(list(eigen_list.cpu().numpy()))
        weight_list_full.append(list(weight_list.cpu().numpy()))

    return eigen_list_full, weight_list_full

def compute_trace(batches, model, loss_f, param_names, n_iters=200, tol=1e-4, window_size=20):
    model.zero_grad()
    params, grads = get_params_grad(model, param_names)
    parameters = params
    
    trace_vhv = []
    trace = 0.

    return_flag = 0
    for i in range(n_iters):
        # 生成 Rademacher 随机向量 {+1, -1}
        v = [torch.randint_like(p, high=2).cuda(parser_args.gpu) * 2 - 1 for p in parameters
            ]
        
        vHv, THv = hv_computation_dataloader(parameters, batches, model, loss_f, v, param_names)

        trace_vhv.append(group_product(THv, v).cpu().item())
        
        if i > 2*window_size:
            trace = np.mean(trace_vhv)
            recent_estimates = trace_vhv[-window_size:]
            moving_avg_change = abs(np.mean(recent_estimates) - trace)
            #print(i, trace, moving_avg_change/abs(trace))
            if moving_avg_change/abs(trace) < tol:
                print(f"converge to {trace}, stop at the {i}-th sampling")
                return trace
        #if abs(np.mean(trace_vhv) - trace) / (trace + 1e-6) < tol:
        #    print(i, np.mean(trace_vhv))
            #return_flag += 1
            #if return_flag > 4:
            #    return np.mean(trace_vhv)
        #else:
        #    trace = np.mean(trace_vhv)

    trace = np.mean(trace_vhv)

    return trace

def compute_power_trace(batches, model, loss_f, param_names, n_iters=200, tol=1e-4, a=1, window_size=20):
    model.zero_grad()
    params, grads = get_params_grad(model, param_names)
    parameters = params
    
    trace_vhv = []
    trace = 0.

    return_flag = 0
    for i in range(n_iters):
        # 生成 Rademacher 随机向量 {+1, -1}
        
        v = [torch.randint_like(p, high=2).cuda(parser_args.gpu) * 2 - 1 for p in parameters
            ]
        v0 = v
                
        for j in range(a):
            vHv, v = hv_computation_dataloader(parameters, batches, model, loss_f, v, param_names)

        trace_vhv.append(group_product(v, v0).cpu().item())
        
        if i > 2*window_size:
            trace = np.mean(trace_vhv)
            recent_estimates = trace_vhv[-window_size:]
            moving_avg_change = abs(np.mean(recent_estimates) - trace)
            #print(i, trace, moving_avg_change/abs(trace))
            if moving_avg_change/abs(trace) < tol:
                print(f"converge to {trace}, stop at the {i}-th sampling")
                return trace
        #if abs(np.mean(trace_vhv) - trace) / (trace + 1e-6) < tol:
        #    print(i, np.mean(trace_vhv))
            #return_flag += 1
            #if return_flag > 4:
            #    return np.mean(trace_vhv)
        #else:
        #    trace = np.mean(trace_vhv)

    trace = np.mean(trace_vhv)

    return trace

def cumpute_trace_within_first_layer(batches, model, loss_f, param_names, n_iters=200, tol=1e-4, window_size=20):
    
    def compute_trace_of_vector(parameters, batches, model, loss_f, param_names, n_iters, tol, window_size):
        trace_vhv = []
        trace = 0.

        return_flag = 0
        for i in tqdm.tqdm(range(n_iters), disable=not True):
            # 生成 Rademacher 随机向量 {+1, -1}
            v = [torch.randint_like(p, high=2).cuda(parser_args.gpu) * 2 - 1 for p in parameters
                ]
            
            vHv, THv = hv_computation_dataloader(parameters, batches, model, loss_f, v, param_names)
            
            vHv = THv[0] * v[0]
        
            trace_vhv.append(vHv)

        trace_list, norm_list = [], []
        tmp_size = parameters[0].size(0)
        for i in range(tmp_size):
            trace = np.mean([torch.sum(item[:,i]).item() for item in trace_vhv])
            trace_list.append(trace)

            param_v = parameters[0][:,i]
            norm_list.append(torch.norm(param_v, p=2).item())
        
        return trace_list, norm_list

    model.zero_grad()
    params, grads = get_params_grad(model, param_names)
    parameters = params

    trace_list, norm_list = compute_trace_of_vector(parameters, batches, model, loss_f, param_names, n_iters, tol, window_size)

    return trace_list, norm_list

def cumpute_trace_within_first_layer_with_v(batches, model, loss_f, param_names, n_iters=200, tol=1e-4, window_size=20):
    
    def compute_trace_of_vector(parameters, batches, model, loss_f, param_names, n_iters, tol, window_size):
        trace_list, norm_list = [], []
        tmp_size = parameters[0].size(0)
        for index in range(tmp_size):
            trace_vhv = []
            for i in tqdm.tqdm(range(n_iters), disable=not True):
                # 生成 Rademacher 随机向量 {+1, -1}
                v = [torch.zeros_like(p).cuda(parser_args.gpu) * 2 - 1 for p in parameters
                    ]
                v[0][:,index] = torch.randint_like(v[0][:,index], high=2).cuda(parser_args.gpu) * 2 - 1

                vHv, THv = hv_computation_dataloader(parameters, batches, model, loss_f, v, param_names)
                
                vHv = group_product(THv, v).cpu().item()

                trace_vhv.append(vHv)
            trace = np.mean(trace_vhv)
            trace_list.append(trace)
            print(i, trace)

        for i in range(tmp_size):
            param_v = parameters[0][:,i]
            norm_list.append(torch.norm(param_v, p=2).item())
        
        return trace_list, norm_list

    model.zero_grad()
    params, grads = get_params_grad(model, param_names)
    parameters = params

    trace_list, norm_list = compute_trace_of_vector(parameters, batches, model, loss_f, param_names, n_iters, tol, window_size)
    print(np.sum(trace_list))
    
    return trace_list, norm_list

def block_eigen(batches, model, loss_f, param_names):
    trace = 0.

    model.zero_grad()
    params, grads = get_params_grad(model, param_names)
    parameters = params

    print(parameters[0].size())
    tmp_size = parameters[0].size(1)

    eigen_list = []
    for i in tqdm.tqdm(range(tmp_size), disable=not True):
        vector = parameters[0][:,i]
        hessian_size = vector.size(0)
        hessian = torch.zeros((hessian_size,hessian_size)).cuda(parser_args.gpu)
        
        batch_sum = 0.
        for i_batch, (inputs, targets) in enumerate(batches):
            for j in tqdm.tqdm(range(vector.size(0)), disable=not False):
                inputs, targets = inputs.cuda(parser_args.gpu), targets.cuda(parser_args.gpu)
                model.zero_grad()
                outputs = model(inputs)
                loss = loss_f(outputs, targets)

                grads = torch.autograd.grad(loss, parameters, create_graph=True)
                grad = grads[0][j,i]
                ggrad = torch.autograd.grad(grad, parameters, create_graph=False)
                hessian_row = ggrad[0][:,i]

                hessian[:,j] += hessian_row*inputs.size(0)
            batch_sum += inputs.size(0)
        hessian /= batch_sum
        eigvals, eigvecs = torch.linalg.eigh(hessian)
        eigen_list.append(eigvals.cpu().numpy())
        trace += np.sum(eigvals.cpu().numpy())
        #print('aadadadada', np.sum(eigvals.cpu().numpy()))
    
    print(trace)

def sample_block_eigen(batches, model, loss_f, param_names, sample_size):
    trace = 0.

    model.zero_grad()
    params, grads = get_params_grad(model, param_names)
    parameters = params

    print(parameters[0].size())
    tmp_size = parameters[0].size(1)

    eigen_list, trace_list, max_eig_list, norm_list = [], [], [], []
    random_numbers = np.random.randint(0, tmp_size+1, size=sample_size)
    for i in tqdm.tqdm(range(tmp_size), disable=not True):
        if i not in random_numbers:
            continue
        vector = parameters[0][:,i]
        hessian_size = vector.size(0)
        hessian = torch.zeros((hessian_size,hessian_size)).cuda(parser_args.gpu)
        
        batch_sum = 0.
        for i_batch, (inputs, targets) in enumerate(batches):
            for j in tqdm.tqdm(range(vector.size(0)), disable=not False):
                inputs, targets = inputs.cuda(parser_args.gpu), targets.cuda(parser_args.gpu)
                model.zero_grad()
                outputs = model(inputs)
                loss = loss_f(outputs, targets)

                grads = torch.autograd.grad(loss, parameters, create_graph=True)
                grad = grads[0][j,i]
                ggrad = torch.autograd.grad(grad, parameters, create_graph=False)
                hessian_row = ggrad[0][:,i]

                hessian[:,j] += hessian_row*inputs.size(0)
            batch_sum += inputs.size(0)
        hessian /= batch_sum
        eigvals, eigvecs = torch.linalg.eigh(hessian)
        eigen_list.append(eigvals.cpu().numpy())
        trace += np.sum(eigvals.cpu().numpy())
        #print('aadadadada', np.sum(eigvals.cpu().numpy()))

        trace_list.append(np.sum(eigvals.cpu().numpy()))
        max_eig_list.append(np.max(eigvals.cpu().numpy()))
        norm_list.append(torch.norm(vector, p=2).item())
        
    print(trace, trace*tmp_size/sample_size)
    trace_norm = [a*b**2 for a, b in zip(trace_list, norm_list)]
    max_eig_norm = [a*b**2 for a, b in zip(max_eig_list, norm_list)]

    return np.sum(trace_norm), np.sum(max_eig_norm)


def input_eigen(model, batches, loss_f, n_iters=100, tol=1e-4, window_size=20):
    trace_vhv = []

    for i in range(n_iters):

        for i_batch, (x, y) in enumerate(batches):
            if i_batch == 0:
                v = [torch.randint_like(x, high=2).cuda(parser_args.gpu) * 2 -1]

        num_data = 0.
        THv = [torch.zeros(p.size()).cuda(parser_args.gpu) for p in v
                ]  # accumulate result
        for i_batch, (inputs, targets) in enumerate(batches):
            inputs, targets = inputs.cuda(parser_args.gpu), targets.cuda(parser_args.gpu)
            inputs.requires_grad = True
            model.zero_grad()
            outputs = model(inputs)
            loss = loss_f(outputs, targets)
            grads = torch.autograd.grad(loss, [inputs], create_graph=True)
            #loss.backward(create_graph=True)
            #params, grads = get_params_grad(model, param_names)
            model.zero_grad()
            
            Hv = torch.autograd.grad(grads, [inputs], grad_outputs=v, only_inputs=True, retain_graph=False)
            THv = [
                    THv1 + Hv1 * float(inputs.size(0)) + 0.
                    for THv1, Hv1 in zip(THv, Hv)
                ]
            num_data += float(inputs.size(0))
            
        THv = [THv1 / float(num_data) for THv1 in THv]
        vHv = group_product(THv, v).cpu().item()
        trace_vhv.append(vHv)

        if i > 2*window_size:
            trace = np.mean(trace_vhv)
            recent_estimates = trace_vhv[-window_size:]
            moving_avg_change = abs(np.mean(recent_estimates) - trace)
            #print(i, trace, moving_avg_change/abs(trace))
            if moving_avg_change/abs(trace) < tol:
                print(f"converge to {trace}, stop at the {i}-th sampling")
                return trace

    return trace


def input_ii(model, batches, loss_f):
    import torch.autograd.functional as F

    if parser_args.dataset == 'cifar10':
        input_dim = 3 * 32 * 32
    elif parser_args.dataset == 'mnist':
        input_dim = 28*28
    else:
        raise ValueError('input dim is not defined!')
    hessian = None
    for i_batch, (inputs, targets) in tqdm.tqdm(enumerate(batches), disable=not False, desc="input hessian"):
        inputs, targets = inputs.cuda(parser_args.gpu), targets.cuda(parser_args.gpu)
        inputs.requires_grad = True
        def loss_fn(inputs):
            outputs = model(inputs)
            if parser_args.batch_size == 1:
                return loss_f(outputs.unsqueeze(0), targets)
            else:
                return loss_f(outputs, targets)
        model.zero_grad()
        outputs = model(inputs)
        #loss = loss_f(outputs, targets)
        hessian_matrix = F.hessian(loss_fn, inputs)
        model.zero_grad()

        hessian_samples = torch.stack([hessian_matrix[i, :, :, :, i, :, :, :] for i in range(parser_args.batch_size)])
        hessian_samples = hessian_samples.contiguous()
        hessian_samples_flat = hessian_samples.view(parser_args.batch_size, input_dim, input_dim)
        hessian_sum = hessian_samples_flat.sum(dim=0)
        hessian_matrix_final = hessian_sum #.view(input_dim, input_dim)
                
        if hessian is None:
            hessian = torch.zeros_like(hessian_matrix_final)
        hessian += hessian_matrix_final

    trace_hessian = torch.trace(hessian)
    eigenvalues = torch.linalg.eigvals(hessian)
    eigenvalues_real = eigenvalues.real
    diagonal_elements = hessian.diagonal()

    #print(trace_hessian, eigenvalues_real, diagonal_elements)

    return trace_hessian.item(), eigenvalues_real.cpu().numpy(), diagonal_elements.cpu().numpy()

def gaussian(x, x0, sigma_squared):
    return np.exp(-(x0 - x)**2 /
                  (2.0 * sigma_squared)) / np.sqrt(2 * np.pi * sigma_squared)

def density_generate(eigenvalues,
                     weights,
                     num_bins=100000,#100000
                     sigma_squared=1e-5, #1e-10,#-10
                     overhead=0.01,
                     file_path=''):

    eigenvalues = np.array(eigenvalues)
    weights = np.array(weights)

    lambda_max = np.mean(np.max(eigenvalues, axis=1), axis=0) + overhead
    lambda_min = np.mean(np.min(eigenvalues, axis=1), axis=0) - overhead

    grids = np.linspace(lambda_min, lambda_max, num=num_bins)
    sigma = sigma_squared * max(1, (lambda_max - lambda_min))

    num_runs = eigenvalues.shape[0]
    #print(num_runs)
    density_output = np.zeros((num_runs, num_bins))

    for i in range(num_runs):
        for j in range(num_bins):
            x = grids[j]
            tmp_result = gaussian(eigenvalues[i, :], x, sigma)
            density_output[i, j] = np.sum(tmp_result * weights[i, :])
    density = np.mean(density_output, axis=0)
    normalization = np.sum(density) * (grids[1] - grids[0])
    density = density / normalization
    #print(len(density),len(grids),sum(density))
    eigen_path = file_path + '/density.txt'
    f = open(eigen_path,"w")
    for line in density:
        f.write(str(line)+'\n')
    f.close()
    grids_path = file_path + '/grids.txt'
    f = open(grids_path,"w")
    for line in grids:
        f.write(str(line)+'\n')
    f.close()
    return density, grids

def get_esd_plot(eigenvalues, weights, file_path, plot=False):
    density, grids = density_generate(eigenvalues, weights, file_path=file_path)
    plt.semilogy(grids, density + 1.0e-7)
    plt.ylabel('Density (Log Scale)', fontsize=14, labelpad=10)
    plt.xlabel('Eigenvlaue', fontsize=14, labelpad=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.axis([np.min(eigenvalues) - 1, np.max(eigenvalues) + 1, None, None])
    plt.tight_layout()
    if plot:
        plt.show()
    #plt.savefig(image_path)

def get_sample_coordinate(tensor_list):
    # 计算所有 tensor 的元素数量
    total_elements = sum(tensor.numel() for tensor in tensor_list)
    
    # 随机选择一个元素的全局索引
    global_idx = torch.randint(0, total_elements, (1,)).item()
    
    # 根据全局索引找到对应的 tensor 和局部索引
    start_idx = 0
    for tensor_idx, tensor in enumerate(tensor_list):
        tensor_num_elements = tensor.numel()
        if global_idx < start_idx + tensor_num_elements:
            local_idx = global_idx - start_idx  # 在该 tensor 中的局部索引
            coordinate = torch.unravel_index(torch.tensor(local_idx), tensor.shape)  # 转换为 Tensor 并获取坐标
            sampled_value = tensor[coordinate]  # 获取该元素的值
            break
        start_idx += tensor_num_elements  # 更新起始索引，进入下一个 tensor
        
    #print(f"采样的数字: {sampled_value}")
    
    return tensor_idx, coordinate

def compute_row_hessain(model, batches, loss_f, param_names):
    num_data, row_l1_norm = 0., 0.
    for i_batch, (inputs, targets) in enumerate(batches):
        inputs, targets = inputs.cuda(parser_args.gpu), targets.cuda(parser_args.gpu)
    
        model.zero_grad()

        outputs = model(inputs)
        loss = loss_f(outputs, targets)
        loss.backward(create_graph=True)
        params, grads = get_params_grad(model, param_names)

        tensor_idx, coordinate = get_sample_coordinate(grads)
        grad = grads[tensor_idx][coordinate]

        hessian_row = torch.autograd.grad(grad,
                                          params,
                                          only_inputs=True,
                                          retain_graph=False)
        
        l1_norm = 0.
        for item in hessian_row:
            l1_norm += torch.norm(item, p=1)
        
        row_l1_norm += l1_norm * inputs.size(0)
        num_data += inputs.size(0)
    
    return (row_l1_norm/num_data).item()

def get_zero_eig_num(model, batches, loss_f, param_names, iter_n=100, tol=1e-5):
    tol_n = 0.
    for i in range(iter_n):
        row_norm = compute_row_hessain(model, batches, loss_f, param_names)
        if row_norm < tol:
            tol_n += 1
    
    return tol_n/iter_n

def read_density(path):
    density_list = []
    density_f = open(path + 'density.txt',"r")
    density_lines = density_f.readlines()
    for density_line in density_lines:
        density = float(density_line[:-1])
        density_list.append(density)
    density_f.close()
    grids_list = []
    grids_f = open(path + 'grids.txt',"r")
    grids_lines = grids_f.readlines()
    for grids_line in grids_lines:
        grids = float(grids_line[:-1])
        grids_list.append(grids)
    grids_f.close()

    return density_list, grids_list

def get_real_density(zero_eig_rate, density_list, grids_list):
    if zero_eig_rate == 0.:
        return density_list, grids_list

    sparse_density_sum = zero_eig_rate/(1-zero_eig_rate) * sum(density_list)
    new_density_list, new_grids_list = [], []
    for w in range(len(density_list)):
        if density_list[w] > 0:
            if w == 0 and grids_list[0] > 0:
                print('add!')
                new_density_list.append(sparse_density_sum)
                new_grids_list.append(1e-30)
            new_density_list.append(density_list[w])
            new_grids_list.append(grids_list[w])
            if grids_list[w]<0 and grids_list[w+1]>0:
                print('add!')
                new_density_list.append(sparse_density_sum)
                new_grids_list.append(1e-30)
    density_list, grids_list = new_density_list, new_grids_list

    return density_list, grids_list

def sample_density(density_list, grids_list, model, param_names, smaple_num=None):
    model.zero_grad()
    params, grads = get_params_grad(model, param_names)
    sample_n = 0
    for item in params:
        tmp = torch.ones_like(item)
        sample_n += torch.norm(tmp, p=1).item()
   
    import random
    density_sum = np.sum(density_list)
    new_density_list = [item/density_sum for item in density_list]

    eig_list = []
    if smaple_num is not None:
        sample_n = smaple_num
    for i in range(int(sample_n)):
        rand_prob = random.uniform(0, 1)
        
        density_array = np.array(new_density_list)
        cumsum = np.cumsum(density_array)
        idx = np.searchsorted(cumsum, rand_prob, side='right')
        sample_eig = grids_list[idx]
        print(idx, sample_eig, np.shape(density_array), len(grids_list))
        eig_list.append(sample_eig)

    return eig_list

def cumpute_gradient(model, batches, loss_f, param_names='model.to_patch_embedding.1.weight'):   
    all_num = 0.
    losses = 0.

    params, grads = get_params_grad(model, param_names)
    grad = [torch.zeros(p.size()).cuda(parser_args.gpu) for p in params
            ]
    for i_batch, (inputs, targets) in enumerate(batches):
        inputs, targets = inputs.cuda(parser_args.gpu), targets.cuda(parser_args.gpu)
    
        model.zero_grad()

        outputs = model(inputs)
        loss = loss_f(outputs, targets)
        loss.backward()
        params, grads = get_params_grad(model, param_names)

        grad = [
                g + g1 * float(inputs.size(0))
                for g, g1 in zip(grad, grads)
            ]
        
        losses += float(inputs.size(0))*loss.item()
        all_num += float(inputs.size(0))

    grad = [g / float(all_num) for g in grad]
    all_grad1, all_grad2 = 0., 0.
    weight_num = 0.
    for item in grad:
        all_grad1 += torch.norm(item, p=1).item() * inputs.size(0)
        all_grad2 += torch.norm(item, p=2).item()**2 * inputs.size(0)
        weight_num += torch.norm(torch.ones_like(item), p=1).item()

    max_abs_val = max(t.abs().max().item() for t in grad)
      
    return all_grad1, all_grad2**0.5, max_abs_val, weight_num#, losses// float(all_num)

def compute_norm(model, param_names='model.to_patch_embedding.1.weight'):
    all_norm = 0.
    params, grads = get_params_grad(model, param_names)
    for item in params:
        all_norm += torch.norm(item).item()**2
    
    return all_norm**0.5

def compute_singular_vlaue(model, param_names='model.to_patch_embedding.1.weight'):
    params, grads = get_params_grad(model, param_names)
    param = params[0]

    singular_values = torch.linalg.svdvals(param)
    #print(singular_values.detach().cpu().numpy())
    
    return singular_values.detach().cpu().numpy()

def get_loss_and_err(model, loss_fn, x, y):
    with torch.no_grad():
        output = model(x)
        loss = loss_fn(output, y)
        err = (output.max(1)[1] != y).float().mean()
    return loss.item(), err.item()