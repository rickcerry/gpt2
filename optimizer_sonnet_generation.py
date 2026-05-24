from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer

@torch.no_grad()
def newton_schultz_5(X, iter=5):
    transposed = False

    if X.size(0) > X.size(1):
        X = X.T
        transposed = True

    X_n = X / (torch.norm(X) + 1e-12)

    for _ in range(iter):
        A = X_n @ X_n.T
        X_n = 3.4445 * X_n - 4.7750 * A @ X_n + 2.0315 * A @ A @ X_n

    if transposed:
        X_n = X_n.T

    return X_n

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
                
                # 1. Retrive all hyperparameters and values from group (otherwise, initialize)
                if "t" not in state:
                    state["t"] = 0
                    state["m_t"] = torch.zeros_like(p.data)
                    state["v_t"] = torch.zeros_like(p.data)
                
                eps = group["eps"]
                b1, b2 = group["betas"]
                weight_decay = group["weight_decay"]

                state["t"] += 1

                m_t = state["m_t"]
                v_t = state["v_t"]
                
                # 1. Update moments
                m_t.mul_(b1).add_(grad, alpha=1 - b1)
                v_t.mul_(b2).addcmul_(grad, grad, value=1 - b2)
                t = state["t"]

                # 2. Efficient bias correction
                alpha_t = alpha * ((1 - b2 ** t) ** 0.5) / (1 - b1 ** t)

                # 3. Weight decay and learning rate
                p.data.add_(p.data, alpha=-alpha * weight_decay)

                # 4. Adam update
                p.data.addcdiv_(m_t, v_t.sqrt().add(eps), value=-alpha_t)



        return loss
    
class NorMuon(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True
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
              if p.dim() == 1:
                      if p.grad is None:
                          continue
                      grad = p.grad.data

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
              else:
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
                      state["v"] = torch.zeros((p.data.shape[0], 1), device=p.device)

                  state["t"] += 1
                  state["M"] = (beta_1 * state["M"]) + ((1 - beta_1) * grad)

                  O_t = newton_schultz_5(state["M"])
                  state["v"] = (beta_2 * state["v"]) + ((1 - beta_2) * torch.mean(O_t*O_t, dim=1, keepdims=True))

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
              if p.dim() == 1:
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
              else:
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
        
class MuonWrapper:
    def __init__(self, model, args):
        self.muon_params = []
        self.adamw_decay_params = []
        self.adamw_no_decay_params = []

        self.muon_names = []
        self.adamw_decay_names = []
        self.adamw_no_decay_names = []

        self.muon = None
        self.adamw = None

        self.split_params(model)

        if len(self.muon_params) > 0:
            self.muon = torch.optim.Muon(
                self.muon_params,
                lr=args.lr,
                momentum=args.beta1,
                weight_decay=args.weight_decay,
            )

        adamw_groups = []

        if len(self.adamw_decay_params) > 0:
            adamw_groups.append({
                "params": self.adamw_decay_params,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            })

        if len(self.adamw_no_decay_params) > 0:
            adamw_groups.append({
                "params": self.adamw_no_decay_params,
                "lr": args.lr,
                "weight_decay": 0.0,
            })

        if len(adamw_groups) > 0:
            self.adamw = torch.optim.AdamW(
                adamw_groups,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
            )

        print(f"Muon params: {len(self.muon_params)}")
        print(f"AdamW decay params: {len(self.adamw_decay_params)}")
        print(f"AdamW no-decay params: {len(self.adamw_no_decay_params)}")

    def split_params(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            is_gpt_hidden_weight = (
                name.startswith("gpt.gpt_layers.")
                and name.endswith(".weight")
                and param.ndim == 2
                and "layer_norm" not in name
            )

            is_bias = name.endswith(".bias")
            is_norm = "layer_norm" in name
            is_embedding = (
                name == "gpt.word_embedding.weight"
                or name == "gpt.pos_embedding.weight"
            )

            if is_gpt_hidden_weight:
                self.muon_params.append(param)
                self.muon_names.append(name)

            elif is_bias or is_norm or is_embedding:
                self.adamw_no_decay_params.append(param)
                self.adamw_no_decay_names.append(name)

            else:
                self.adamw_decay_params.append(param)
                self.adamw_decay_names.append(name)

    def step(self):
        if self.muon is not None:
            self.muon.step()

        if self.adamw is not None:
            self.adamw.step()

    def zero_grad(self):
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=True)

        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=True)

    def state_dict(self):
        return {
            "muon": self.muon.state_dict() if self.muon is not None else None,
            "adamw": self.adamw.state_dict() if self.adamw is not None else None,
        }

    def load_state_dict(self, state_dict):
        if self.muon is not None and state_dict.get("muon") is not None:
            self.muon.load_state_dict(state_dict["muon"])

        if self.adamw is not None and state_dict.get("adamw") is not None:
            self.adamw.load_state_dict(state_dict["adamw"])