import math
import torch

from torch import nn


class QFormerLayer(nn.Module): 
    """
    Implements a Query Former Layer that performs cross-attention between learned queries and source embeddings.
    
    This layer uses a set of learned queries to attend to source embeddings, effectively extracting
    relevant information from the source embeddings based on the learned queries. It's designed to
    bridge different modalities by allowing attention between a fixed learned query set and variable
    source embeddings (e.g., from an LLM).
    
    Args:
        d_model (int): Dimension of the target/query embeddings
        n_heads (int): Number of attention heads
        d_keys (int, optional): Dimension of key/query vectors per head. Defaults to d_model // n_heads
        d_llm (int): Dimension of the source/value embeddings (from LLM)
        attention_dropout (float): Dropout rate applied to attention scores. Default: 0.1
        query_length (int): Length of the learned query sequence. Default: 12
    """
    def __init__(
        self, 
        d_model, 
        n_heads, 
        d_keys=None, 
        d_llm=None, 
        attention_dropout=0.1,
        query_length=12
    ):
        super(QFormerLayer, self).__init__()

        self.n_heads = n_heads
        d_keys = d_keys or (d_model // n_heads)
        # Initialize learned queries that will attend to the source embeddings
        self.learned_query = nn.Parameter(torch.zeros(query_length, d_model))

        # Linear projections for query, key, value, and output
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, source_embedding, value_embedding):
        """
        Performs cross-attention between learned queries and source embeddings.
        
        Applies multi-head attention mechanism where learned queries attend to source embeddings,
        and applies the resulting attention weights to value embeddings.
        
        Args:
            source_embedding (torch.Tensor): Source embeddings used as keys [S, d_llm], 
                                             where S is the source sequence length
            value_embedding (torch.Tensor): Value embeddings used for values [S, d_llm],
                                            where S is the source sequence length
        
        Returns:
            torch.Tensor: Output tensor after applying attention and projection [1, L, d_llm],
                          where L is the query length
        """
        # target_embedding: [L, D] - learned queries that will attend to source
        # source_embedding/value_embedding: [S, D'] - source representations to attend to
        target_embedding = self.learned_query
        L, _ = target_embedding.shape  # L: query length, _: embedding dimension
        S, _ = source_embedding.shape  # S: source length, _: embedding dimension
        H = self.n_heads               # H: number of attention heads

        # Project embeddings to multi-head format
        # Transform learned queries to query space
        target_embedding = self.query_projection(target_embedding).view(L, H, -1)   # [L, H, d_head]
        # Transform source embeddings to key space
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)      # [S, H, d_head]
        # Transform value embeddings to value space
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)      # [S, H, d_head]

        # Compute attention scores between queries and keys
        scale = 1. / math.sqrt(target_embedding.shape[-1])  # Scale factor for attention
        # Calculate attention scores using Einstein summation: "lhe,she->hls"
        # l: query position, s: source position, h: head, e: embedding dimension
        scores = torch.einsum("lhe,she->hls", target_embedding, source_embedding)   # [H, L, S]
        # Apply softmax and dropout to get attention weights
        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        # Apply attention weights to values
        out = torch.einsum("hls,she->lhe", A, value_embedding)                      # [L, H, d_head]
        out = out.reshape(L, -1)  # merge heads: [L, H*d_head]

        # Project output back to desired dimension and add batch dimension
        return self.out_projection(out).unsqueeze(0)  # [1, L, d_llm]
