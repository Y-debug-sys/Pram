import torch
from torch import nn
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .modules.context import QFormerLayer
from .modules.head_network import PolicyHead
from .modules.bf_wrapper import DtypeAlignWrapper
from .modules.lora import truncate_and_inject_lora
from .prompt import prepare_prompt_by_group, prepare_prompt_by_source


class PramModel(nn.Module):
    """
    PRAM Model for multi-commodity flow planning using multimodal language models.
    
    This model combines vision-language models with specialized components to solve 
    multi-commodity flow problems by leveraging visual and textual inputs to make
    decisions about routing in network topologies.
    
    Args:
        dim_mllm (int): Dimension of the multimodal language model
        num_nodes (int): Number of nodes in the network topology
        num_paths (int): Number of candidate paths between each node pair
        model_name (str): Name of the base model ('qwen-3b' or 'qwen-7b')
        num_layers (int): Number of layers to inject LoRA into (default 8)
        dim_hidden (int): Hidden dimension for internal processing (default 128)
        mcf_objective (str): Multi-commodity flow objective ('mlu', 'mtf', or 'mcf')
        len_context (int, optional): Length of context for QFormer (defaults to num_nodes)
        dropout (float): Dropout rate for context encoder (default 0.1)
        lora_rank (int): Rank for LoRA adaptation (default 16)
        lora_alpha (int): Alpha parameter for LoRA scaling (default 32)
        lookup_size (int): Size of the lookup table for embeddings (default 1000)
        num_agents (int, optional): Number of agents for distributed processing
    """

    def __init__(
        self,
        dim_mllm,
        num_nodes,
        num_paths,
        model_name='qwen',
        num_layers=8,
        dim_hidden=128,
        mcf_objective='mlu',
        len_context=None,
        dropout=0.1,
        lora_rank=16,
        lora_alpha=32,
        lookup_size=1000,
        num_agents=None
    ):
        super(PramModel, self).__init__()

        self.objective = mcf_objective
        self.num_nodes, self.num_paths = num_nodes, num_paths
        self.num_agents, self.num_layers = num_agents, num_layers
        len_context = num_nodes if len_context is None else len_context 
        len_context = max(48, min(len_context, num_nodes)) 
        self.output_size = num_nodes * (num_nodes - 1) * num_paths

        """ Load VL model and processor. """
        if model_name == 'qwen3b':
            model_path = "./mlms/Qwen2.5-VL-3B-Instruct"
        elif model_name == 'qwen7b':
            model_path = "./mlms/Qwen2.5-VL-7B-Instruct"
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
        
        self.mllm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                dtype="auto",
                device_map=None, 
                low_cpu_mem_usage=True  
            )
    
        self.processor = AutoProcessor.from_pretrained(model_path)

        for param in self.mllm_model.parameters():
            param.requires_grad = False

        # Inject LoRA into the first num_layers layers of the language model
        self.mllm_model = truncate_and_inject_lora(self.mllm_model, r=lora_rank, alpha=lora_alpha, num_layers=num_layers)
        
        # self.mllm_model.enable_input_require_grads() 
        # self.mllm_model.gradient_checkpointing_enable()

        # embedding related
        self.word_embeddings = self.mllm_model.get_input_embeddings().weight
        self.word_embeddings.requires_grad = False
        self.vocab_size = self.word_embeddings.shape[0]
        self.mapping_layer = DtypeAlignWrapper(nn.Linear(self.vocab_size, lookup_size))

        # context encoder
        self.flatten = nn.Flatten(start_dim=-2)
        self.context_embedding = QFormerLayer(
            dim_hidden, n_heads=4, d_keys=dim_hidden, d_llm=dim_mllm,
            attention_dropout=dropout, query_length=len_context
        )

        if num_agents is not None:
            assert self.num_agents < self.num_nodes, "num_agents must be less than num_nodes"
            chunk_size = (num_nodes + num_agents - 1) // num_agents
            s = chunk_size
        else:
            s = 1

        self.llm_out_dim = dim_mllm
        self.head = PolicyHead(
            self.llm_out_dim,
            hidden_size=dim_hidden,
            output_size=int(s * (num_nodes - 1) * num_paths),
            std=0.5,
            obj=mcf_objective
        )

    def forward(
        self,
        sub_images: list,          
        demands: torch.Tensor,      
        fake_data: bool = False,
        test: bool = False
    ):
        """
        Forward pass through the PRAM model.
        
        Args:
            sub_images: List of images (batched) or list[list[images]] representing network topology
            demands: Tensor with shape [B, T, num_nodes] representing flow demands over time
            fake_data: Boolean indicating whether to treat data as having uniform capacity
            test: Boolean indicating whether in test mode (deterministic) or training mode (stochastic)
        
        Returns:
            tuple: A tuple containing:
                - weights (torch.Tensor): Path weights for routing [B, output_size]
                - log_probability (torch.Tensor): Log probabilities of actions [B, output_size]
        """

        B, T, N = demands.shape

        # mean: [B, N], last: [B, N]
        mean = torch.mean(demands, dim=1)
        last = demands[:, -1, :]
        prompts, images = [], []

        for i in range(B):
            images += sub_images

            if self.num_agents is None:
                prompts += prepare_prompt_by_source(
                        last[i], self.num_nodes, self.num_paths, self.objective,
                        history_mean=mean[i], same_cap=fake_data
                    )
            else:
                prompts += prepare_prompt_by_group(
                        last[i], self.num_nodes, self.num_agents, self.num_paths, self.objective,
                        history_mean=mean[i], same_cap=fake_data
                    )

        with torch.no_grad():
            prompt_inputs = self.processor(
                text=prompts,
                images=images,         
                return_tensors="pt",
                padding=True
            )
            prompt_ids = prompt_inputs["input_ids"].to(last.device)
            prompt_attention_mask = prompt_inputs["attention_mask"].to(last.device)
            pixel_values = prompt_inputs["pixel_values"].to(last.device)
            image_grid_thw = prompt_inputs["image_grid_thw"].to(last.device)

            prompt_embeddings = self.mllm_model.get_input_embeddings()(prompt_ids)

        protype = self.mapping_layer(self.word_embeddings.permute(1, 0).contiguous()).permute(1, 0).contiguous()
        context = self.context_embedding(protype, protype).repeat(B * self.num_nodes, 1, 1) 

        llama_enc_out = torch.cat([context, prompt_embeddings], dim=1).contiguous()

        attention_mask = torch.ones((llama_enc_out.shape[0], llama_enc_out.shape[1]), device=llama_enc_out.device)
        attention_mask[:, context.shape[1]:] = prompt_attention_mask
        attention_mask = attention_mask.contiguous()

        dec_out = self.mllm_model(
            inputs_embeds=llama_enc_out,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True
        )

        head_input = self.flatten(dec_out.hidden_states[-1])[:, - self.llm_out_dim:]

        weights, log_probability = self.head.evaluate(head_input, test)

        if log_probability is not None:
            log_probability = log_probability.reshape(B, -1).float().contiguous()

        weights = weights.reshape(B, -1).float().contiguous()
        return weights[:, :self.output_size], log_probability
