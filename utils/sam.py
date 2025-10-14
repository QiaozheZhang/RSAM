import torch
from collections import defaultdict

class ASAM:
    def __init__(self, optimizer, model, rho=0.5, eta=0.01):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)

    @torch.no_grad()
    def ascent_step(self):
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if 'weight' in n:
                t_w[...] = p[...]
                t_w.abs_().add_(self.eta)
                p.grad.mul_(t_w)
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if 'weight' in n:
                p.grad.mul_(t_w)
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(self.rho / wgrad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()


class SAM(ASAM):
    @torch.no_grad()
    def ascent_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            # print(self.rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

class FSAM(ASAM):
    def __init__(self, optimizer, model, rho=0.5, eta=1.0):
        super().__init__(optimizer, model, rho=rho, eta=eta)

    @torch.no_grad()
    def ascent_step(self):
        # fisher_diag_fn(p) -> 与 p 同形的对角 Fisher 估计; 若为 None 就用当前批 g^2
        gtFinv_g = 0.0
        for p in self.model.parameters():
            if p.grad is None: continue
            g = p.grad.detach()
            f = g*g
            e = g / (1.0 + self.eta * f)
            Finv_g = g / (1.0 + self.eta * f)
            self.state[p]["Finv_g"] = Finv_g
            gtFinv_g += (g * Finv_g).sum().item()
        denom = (gtFinv_g ** 0.5) + 1e-16
        scale = self.rho / denom
        for p in self.model.parameters():
            if p.grad is None: 
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.empty_like(p)
                self.state[p]["eps"] = eps
            eps.copy_(self.state[p]["Finv_g"]).mul_(scale)
            p.add_(eps)
        self.optimizer.zero_grad()

class RSAM:
    def __init__(self, optimizer, model, rho=0.5, eta=2, reverse=False):
        self.alpha = eta
        self.optimizer = optimizer
        self.model = model
        self.old_rho = (1-self.alpha)/abs(1-self.alpha)*rho
        self.state = defaultdict(dict)
        self.reverse = reverse

    @torch.no_grad()
    def ascent_step(self):
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2)**2 + 1.e-16
        all_wgrad = torch.stack(wgrads)
        wgrad_power_norm = torch.norm(all_wgrad**self.alpha, p=2)**2 + 1.e-16
        if self.reverse:
            self.rho = -self.old_rho * (wgrad_norm**(self.alpha+1)) / wgrad_power_norm
        else:
            self.rho = self.old_rho * wgrad_power_norm / (wgrad_norm**(self.alpha+1))
        # print(self.rho)

        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(-self.rho)
            p.add_(eps)
        self.optimizer.zero_grad()
        

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()

class NEWRSAM:
    def __init__(self, optimizer, model, rho=0.5, eta=2, reverse=False):
        self.alpha = eta
        self.optimizer = optimizer
        self.model = model
        self.old_rho = (1-self.alpha)/abs(1-self.alpha)*rho
        self.state = defaultdict(dict)
        self.reverse = reverse

    @torch.no_grad()
    def ascent_step(self):
        wgrads = []
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            wgrads.append(torch.norm(p.grad, p=2))
            grads.append(p.grad.flatten())
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2)**2 + 1.e-16
        all_grads = torch.cat(grads)
        wgrad_power_norm = (all_grads.abs()**self.alpha).pow(2).sum() + 1e-16
        if self.reverse:
            self.rho = -self.old_rho * (wgrad_norm**(self.alpha+1)) / wgrad_power_norm
        else:
            self.rho = self.old_rho * wgrad_power_norm / (wgrad_norm**(self.alpha+1))
        # print(self.rho)

        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(-self.rho)
            p.add_(eps)
        self.optimizer.zero_grad()
        

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()