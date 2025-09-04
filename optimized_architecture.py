#!/usr/bin/env python3
"""
Architecture Super Neuron optimisée avec connexions résiduelles et attention améliorée
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class OptimizedMultiHeadAttention(nn.Module):
    """Attention multi-têtes optimisée avec scaling et dropout"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim doit être divisible par num_heads"
        
        # Projections avec initialisation Xavier
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Initialisation Xavier
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Projections Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention avec scaling
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape et projection finale
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out_proj(attn_output)

class OptimizedSpecializedBranch(nn.Module):
    """Branche spécialisée optimisée avec connexions résiduelles"""
    def __init__(self, embed_dim, specialty_type="general", dropout=0.15):
        super().__init__()
        self.specialty_type = specialty_type
        self.embed_dim = embed_dim
        
        # Couche d'entrée
        self.input_norm = nn.LayerNorm(embed_dim)
        
        if specialty_type == "syntax":
            # Structure grammaticale avec attention
            num_heads = max(1, embed_dim // 64)
            self.processor = OptimizedMultiHeadAttention(embed_dim, num_heads, dropout)
            self.feedforward = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim)
            )
            
        elif specialty_type == "semantic":
            # Sens et contexte avec couches denses
            self.processor = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 3),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 3, embed_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(embed_dim * 2, embed_dim)
            )
            self.feedforward = nn.Identity()  # Pas de feedforward supplémentaire
            
        elif specialty_type == "conversational":
            # Dialogue et interaction
            self.processor = nn.LSTM(embed_dim, embed_dim, batch_first=True, dropout=dropout)
            self.feedforward = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim)
            )
            
        else:  # general
            # Traitement général optimisé
            num_heads = max(1, embed_dim // 32)
            self.processor = OptimizedMultiHeadAttention(embed_dim, num_heads, dropout)
            self.feedforward = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim)
            )
        
        # Normalisations pour connexions résiduelles
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Connexion résiduelle 1 : input normalization + processor
        normed_x = self.input_norm(x)
        
        if self.specialty_type == "conversational":
            processed, _ = self.processor(normed_x)
        else:
            processed = self.processor(normed_x)
        
        x = x + self.dropout(processed)  # Connexion résiduelle
        x = self.norm1(x)
        
        # Connexion résiduelle 2 : feedforward
        if not isinstance(self.feedforward, nn.Identity):
            ff_out = self.feedforward(x)
            x = x + self.dropout(ff_out)  # Connexion résiduelle
            x = self.norm2(x)
        
        return x

class OptimizedSuperNeuronConversational(nn.Module):
    """Super Neuron optimisé pour conversations avec routage intelligent"""
    def __init__(self, vocab_size, embed_dim=128, max_seq_length=150, num_branches=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        self.num_branches = num_branches
        
        # Embeddings avec positional encoding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embed_dim)
        self.embedding_dropout = nn.Dropout(0.1)
        
        # Branches spécialisées optimisées
        self.branches = nn.ModuleList([
            OptimizedSpecializedBranch(embed_dim, "syntax", dropout=0.1),
            OptimizedSpecializedBranch(embed_dim, "semantic", dropout=0.15),  
            OptimizedSpecializedBranch(embed_dim, "conversational", dropout=0.12)
        ])
        
        # Routeur intelligent avec attention
        self.router = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_branches),
            nn.Softmax(dim=-1)
        )
        
        # Couche de fusion optimisée
        self.fusion = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # Tête de prédiction finale
        self.output_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, vocab_size)
        )
        
        # Initialisation des poids
        self._init_weights()
    
    def _init_weights(self):
        """Initialisation Xavier pour une convergence stable"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings avec position
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        x = self.embedding_dropout(token_emb + pos_emb)
        
        # Routage intelligent (moyenne sur la séquence pour décision globale)
        route_input = x.mean(dim=1)  # [batch_size, embed_dim]
        routing_weights = self.router(route_input)  # [batch_size, num_branches]
        
        # Traitement par chaque branche
        branch_outputs = []
        for i, branch in enumerate(self.branches):
            branch_out = branch(x)  # [batch_size, seq_len, embed_dim]
            branch_outputs.append(branch_out)
        
        # Fusion pondérée avec les poids de routage
        fused_output = torch.zeros_like(x)
        for i, branch_out in enumerate(branch_outputs):
            weight = routing_weights[:, i:i+1].unsqueeze(-1)  # [batch_size, 1, 1]
            fused_output += weight * branch_out
        
        # Couche de fusion finale
        fused_output = self.fusion(fused_output)
        
        # Prédiction finale
        logits = self.output_head(fused_output)
        
        return logits
    
    def get_routing_info(self, input_ids):
        """Retourne les poids de routage pour analyse"""
        with torch.no_grad():
            batch_size, seq_len = input_ids.shape
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            token_emb = self.token_embedding(input_ids)
            pos_emb = self.position_embedding(positions)
            x = self.embedding_dropout(token_emb + pos_emb)
            
            route_input = x.mean(dim=1)
            routing_weights = self.router(route_input)
            
            return routing_weights.cpu().numpy()
