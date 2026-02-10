import torch
import torch.nn as nn

from .bf_wrapper import DtypeAlignWrapper


def get_parent_module(model: nn.Module, module_name: str):
    """
    Given a module name from named_modules(), return:
        - parent_module: the nn.Module that contains this submodule
        - attr_name: the attribute name of this submodule in parent_module
    """
    names = module_name.split(".")
    parent = model
    # Traverse until the second last name
    for name in names[:-1]:
        parent = getattr(parent, name)
    attr_name = names[-1]
    return parent, attr_name


# -------------------------
# LoRA for nn.Linear
# -------------------------
class LoRALinear(nn.Module):
    """Wrap an existing nn.Linear with LoRA adapters and expose weight/bias."""
    def __init__(self, orig_linear: nn.Linear, r=8, alpha=16):
        super().__init__()
        in_features = orig_linear.in_features
        out_features = orig_linear.out_features
        bias = orig_linear.bias is not None

        # Frozen base
        self.base = nn.Linear(in_features, out_features, bias=bias)
        with torch.no_grad():
            self.base.weight.copy_(orig_linear.weight.data)
            if bias:
                self.base.bias.copy_(orig_linear.bias.data)
        for p in self.base.parameters():
            p.requires_grad = False

        self.weight = self.base.weight
        self.bias = self.base.bias if bias else None

        # LoRA adapters
        self.r = r
        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.scaling = float(alpha) / r
            nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
            nn.init.zeros_(self.lora_B.weight)
            
            self.lora_A.weight.requires_grad = True
            self.lora_B.weight.requires_grad = True
        else:
            self.lora_A = None
            self.lora_B = None
            self.scaling = 0.0

    def forward(self, x):
        base_out = self.base(x)
        if self.r > 0:
            lora_out = self.lora_B(self.lora_A(x)) * self.scaling
            return base_out + lora_out
        else:
            return base_out


# -------------------------
# Truncate and inject LoRA
# -------------------------
def truncate_and_inject_lora(full_model, num_layers, r=8, alpha=16):
    """
    Truncate language and vision layers and inject LoRA:
      - Language: first `num_layers` layers, LoRA on all q_proj/v_proj
      - Vision: first `num_layers` blocks, LoRA only on last block's attention q_proj/v_proj

    Args:
        full_model: Qwen2_5_VLForConditionalGeneration instance
        num_layers (int): number of layers to keep
        r (int): LoRA rank
        alpha (int): LoRA scaling factor
    """
    # -------------------------
    # Truncate language model
    # -------------------------
    lm = full_model.model.language_model
    total_lm_layers = len(lm.layers)
    if num_layers > total_lm_layers:
        raise ValueError(f"num_layers={num_layers} > language_model total={total_lm_layers}")
    lm.layers = nn.ModuleList(lm.layers[:num_layers])
    # print(f"[Truncate] language_model.layers: kept {num_layers}/{total_lm_layers} layers")

    # Inject LoRA into all LM layers' q_proj and v_proj
    for i, layer in enumerate(lm.layers):
        attn = layer.self_attn
        if isinstance(attn.q_proj, nn.Linear):
            attn.q_proj = DtypeAlignWrapper(LoRALinear(attn.q_proj, r=r, alpha=alpha))
            # print(f"[LoRA] LM layer {i}: q_proj -> LoRALinear(r={r}, alpha={alpha})")
        if isinstance(attn.v_proj, nn.Linear):
            attn.v_proj = DtypeAlignWrapper(LoRALinear(attn.v_proj, r=r, alpha=alpha))
            # print(f"[LoRA] LM layer {i}: v_proj -> LoRALinear(r={r}, alpha={alpha})")

    # print("[LoRA] Truncate and injection complete ✅")
    return full_model
