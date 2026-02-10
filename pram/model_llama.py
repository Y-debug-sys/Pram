import torch
import numpy as np

from torch import nn
from .modules.context import QFormerLayer
from .modules.head_network import PolicyHead
from .modules.lora import LoRALinear, get_parent_module
from transformers import MllamaConfig, AutoProcessor, MllamaForConditionalGeneration


class PramModel(nn.Module): 

    def __init__(
        self, 
        dim_mllm,
        num_nodes,
        num_paths,
        model_name='llama',
        num_layers=8,
        dim_hidden=128,
        mcf_objective='mlu',
        len_context=None,
        dropout=0.1,
        lora_rank=16,
        lora_alpha=32,
        lookup_size=1000
    ):
        super(PramModel, self).__init__()

        self.objective = mcf_objective
        self.num_nodes, self.num_paths = num_nodes, num_paths
        len_context = num_nodes if len_context is None else len_context

        """ Load LLM model and tokenizer. """

        if model_name == 'llama':
            model_id = 'meta-llama/Llama-3.2-11B-Vision'  # Model ID for LLaMA 3 8B model 
            model_dir = './mlms/Llama-3.2-11B-Vision'  # Directory to save the model locally
            tokenizer_dir = './mlms/Llama-3.2-11B-Vision/tokenizer.json'  # Directory to save the tokenizer locally

        self.mllm_config = MllamaConfig.from_pretrained(model_dir)
        self.mllm_config.num_hidden_layers = num_layers
        self.mllm_config.output_attentions = True
        self.mllm_config.output_hidden_states = True

        self.processor = AutoProcessor.from_pretrained(model_dir)

        try:
            self.mllm_model = MllamaForConditionalGeneration.from_pretrained(
                model_dir,
                config=self.mllm_config,
                # load_in_4bit=True,
                # torch_dtype=torch.bfloat16
            )
        except EnvironmentError:  # downloads model from HF if not already done
            print("Local model files not found. Attempting to download...")
            self.mllm_model = MllamaForConditionalGeneration.from_pretrained(
                model_id,
                trust_remote_code=True,
                local_files_only=False,
                config=self.mllm_config,
                # load_in_4bit=True
            )
        
        for param in self.mllm_model.parameters():
            param.requires_grad = False  # freeze LLM model

        for name, module in self.mllm_model.named_modules():
            if isinstance(module, nn.Linear) and name.endswith("q_proj"):
                module_parent, attr_name = get_parent_module(self.mllm_model, name)
                setattr(module_parent, attr_name, LoRALinear(module.weight, r=lora_rank, alpha=lora_alpha))

            if isinstance(module, nn.Linear) and name.endswith("v_proj"):
                module_parent, attr_name = get_parent_module(self.mllm_model, name)
                setattr(module_parent, attr_name, LoRALinear(module.weight, r=lora_rank, alpha=lora_alpha))

        self.word_embeddings = self.mllm_model.get_input_embeddings().weight
        self.word_embeddings.requires_grad = False 
        self.vocab_size = self.word_embeddings.shape[0]
        self.mapping_layer = nn.Linear(self.vocab_size, lookup_size)

        self.flatten = nn.Flatten(start_dim=-2)  # (b, s, d) -> (b * s, d)
        self.context_embedding = QFormerLayer(dim_hidden, n_heads=4, d_keys=dim_hidden, d_llm=dim_mllm, 
                                              attention_dropout=dropout, query_length=len_context)
        
        self.llm_out_dim = dim_mllm
        self.head = PolicyHead(
            self.llm_out_dim,
            hidden_size=dim_hidden,
            output_size=(num_nodes - 1) * num_paths,
            std=0.5,
            obj=mcf_objective
        )

    def forward(
        self, 
        sub_images: list,
        demand: np.ndarray, 
        fake_data: bool = False,
        test: bool = False
    ):
        mean = torch.mean(demand, dim=0)
        last = demand[-1, :]
        prompts = prepare_llama_prompt(last, self.num_nodes, self.num_paths, self.objective,
                                       history_mean=mean, same_cap=fake_data)
        
        prompt_inputs = self.processor(text=prompts, return_tensors="pt", padding=True)
        prompt_ids = prompt_inputs["input_ids"].to(last.device) 
        prompt_attention_mask = prompt_inputs["attention_mask"].to(last.device)

        prompt_embeddings = self.mllm_model.get_input_embeddings()(prompt_ids)  # shape: (num_nodes, prompt_token, dim)
        
        protype = self.mapping_layer(self.word_embeddings.permute(1, 0).contiguous()).permute(1, 0).contiguous()
        context = self.context_embedding(protype, protype).repeat(self.num_nodes, 1, 1)
        
        with torch.no_grad():
            image_inputs = self.processor(images=sub_images, return_tensors="pt", size={"height": 32, "width": 32})
            pixel_values = image_inputs["pixel_values"].to(last.device)   # [B, C, H, W]
            pixel_values = pixel_values.transpose(0, 1).contiguous()

            aspect_ratio_ids = torch.ones(pixel_values.shape[0], pixel_values.shape[1], dtype=torch.long, device=pixel_values.device)
            aspect_ratio_mask = torch.ones(pixel_values.shape[:3], dtype=torch.long, device=pixel_values.device)

        llama_enc_out = torch.cat([context, prompt_embeddings], dim=1).contiguous()
        attention_mask = torch.ones((llama_enc_out.shape[0], llama_enc_out.shape[1])).to(llama_enc_out.device)
        attention_mask[:, context.shape[1]:] = prompt_attention_mask
        attention_mask = attention_mask.contiguous()

        dec_out = self.mllm_model(
            inputs_embeds=llama_enc_out,                 
            attention_mask=attention_mask,        
            pixel_values=pixel_values,    
            aspect_ratio_ids=aspect_ratio_ids,               
            output_hidden_states=True,
            aspect_ratio_mask=aspect_ratio_mask
        )
        head_input = self.flatten(dec_out.hidden_states[-1])[:, - self.llm_out_dim:] 

        weights, log_probability = self.head.evaluate(head_input, test)
        if log_probability is not None: log_probability.reshape(1, -1).float().contiguous()
        
        return weights.reshape(1, -1).float().contiguous(), log_probability
