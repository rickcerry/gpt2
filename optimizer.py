from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary.
                alpha = group["lr"]
                
                beta_1, beta_2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                correct_bias = group["correct_bias"]

                if state.get("t") is None:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                state["t"] += 1
                state["m"] = (beta_1 * state["m"]) + ((1 - beta_1) * grad)
                state["v"] = (beta_2 * state["v"]) + ((1 - beta_2) * torch.square(grad))

                if correct_bias:
                    alpha_t = alpha * (1 - beta_2**state["t"])**0.5 / (1 - beta_1**state["t"])
                else:
                    alpha_t = alpha

                p.data = p.data - alpha_t * state["m"] / (state["v"]**0.5 + eps)
                p.data = p.data - alpha * weight_decay * p.data

                ### TODO: Complete the implementation of AdamW here, reading and saving
                ###       your state in the `state` dictionary above.
                ###       The hyperparameters can be read from the `group` dictionary
                ###       (they are lr, betas, eps, weight_decay, as saved in the constructor).
                ###
                ###       To complete this implementation:
                ###       1. Update the first and second moments of the gradients.
                ###       2. Apply bias correction
                ###          (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                ###          also given in the pseudo-code in the project description).
                ###       3. Update parameters (p.data).
                ###       4. Apply weight decay after the main gradient-based updates.
                ###
                ###       Refer to the default project handout for more details.
                ### YOUR CODE HERE



                #raise NotImplementedError


        return loss
    


@torch.no_grad()
def newton_schultz_5(X, iter=5):

  transposed = False
  if X.size(0) > X.size(1):
      X = X.T
      transposed = True
  orig_dtype = X.dtype

  X_n = X / (torch.norm(X) + 1e-12)

  X_n = X_n.bfloat16()
  for _ in range(iter):
    A = X_n@X_n.T
    B = A@X_n
    C = A@B
    X_n = 3.4445*X_n - 4.7750*B + 2.0315*C

  if transposed:
    X_n= X_n.T

  return X_n.to(orig_dtype)



class NorMuon(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            ns_steps: int = 5
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)
    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
          for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad.data
            state = self.state[p]

            eta = group["lr"]
            beta_1, beta_2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]

            if state.get("t") is None:
                state["t"] = 0
                state["M"] = torch.zeros_like(p.data, device=p.device)
                state["v"] = torch.zeros((p.data.shape[0], 1), device=p.device)

            state["t"] += 1
            state["M"].mul_(beta_1).add_(grad, alpha=1-beta_1)

            O_t = newton_schultz_5(state["M"], iter=ns_steps)
            state["v"].mul_(beta_2).add_(torch.mean(O_t**2, dim=1, keepdims=True), alpha=1-beta_2)

            O_t_hat = O_t / (torch.sqrt(state["v"]) + eps)

            m, n = p.data.shape
            eta_hat = 0.2 * eta * math.sqrt(m * n) / (torch.norm(O_t_hat) + eps)

            p.data = p.data - eta * weight_decay * p.data - eta_hat * O_t_hat

        return loss




class HTMuon(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            p_: float = 0.5 #NEW PARAMETER (BETWEEN 0 and 1!)
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        if not 0.0<= p_ <= 1:
          raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias = correct_bias, p_=p_)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
              if p.grad is None:
                  continue
              grad = p.grad.data

              # State should be stored in this dictionary.
              state = self.state[p]

              # Access hyperparameters from the `group` dictionary.
              eta = group["lr"]
              beta_1, beta_2 = group["betas"]
              eps = group["eps"]
              weight_decay = group["weight_decay"]
              correct_bias = group["correct_bias"]

              p_ = group["p_"]

              if state.get("t") is None:
                  state["t"] = 0
                  state["M"] = torch.zeros_like(p.data, device=p.device)

              state["t"] += 1

              state["M"] = (beta_1 * state["M"]) + ((1 - beta_1) * grad)

              U, Sig, V = torch.linalg.svd(state["M"], full_matrices=False)
              O = (U * (Sig**p_).unsqueeze(0)) @ V


              m = p.data.shape[0]
              n = p.data.shape[1]
              s = math.sqrt(max(1, m/n))

              p.data = p.data - eta * weight_decay*p.data - eta*O*s

        return loss
    

class Muon_2(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias = correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
              if p.dim() !=2:
                raise ValueError("Parameter is not 2d as it should be, instead it is {}d".format(p.dim()))
              if p.grad is None:
                  continue
              grad = p.grad.data
              state = self.state[p]

              eta = group["lr"]
              beta_1, beta_2 = group["betas"]
              eps = group["eps"]
              weight_decay = group["weight_decay"]

              if state.get("t") is None:
                  state["t"] = 0
                  state["M"] = torch.zeros_like(p.data, device=p.device)
                  state["V"] = torch.zeros_like(p.data, device=p.device)

              state["t"] += 1
              state["M"] = (beta_1 * state["M"]) + ((1 - beta_1) * grad)
              state["V"] = (beta_2 * state["V"]) + ((1 - beta_2) * (grad * grad))

              M_tilde = state["M"] / (torch.sqrt(state["V"]) + eps)
              O = newton_schultz_5(M_tilde)

              m, n = p.data.shape
              p.data = p.data - eta * math.sqrt(n / m) * O
              p.data = p.data - eta * weight_decay * p.data

            return loss
