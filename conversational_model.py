#!/usr/bin/env python3
"""
Super Neuron Conversationnel - Optimisé pour le dialogue naturel
Architecture spécialisée dans l'apprentissage de conversations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import pickle
import time
from tqdm import tqdm
from collections import Counter

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim doit être divisible par num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Projections Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape et projection finale
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out_proj(attn_output)

class ConversationalPreprocessor(nn.Module):
    """Pré-processeur spécialisé pour le dialogue conversationnel"""
    def __init__(self, embed_dim, specialty_type):
        super().__init__()
        self.specialty_type = specialty_type
        self.embed_dim = embed_dim
        
        if specialty_type == "dialog_flow":
            # Focus sur le flux de conversation
            dialog_heads = max(1, embed_dim // 32)
            self.processor = nn.MultiheadAttention(embed_dim, dialog_heads, batch_first=True)
            self.norm = nn.LayerNorm(embed_dim)
            
        elif specialty_type == "sentiment":
            # Focus sur le ton et l'émotion
            self.processor = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),  # Activation plus douce pour les émotions
                nn.Dropout(0.1),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Tanh()  # Normalise les émotions
            )
            self.norm = nn.LayerNorm(embed_dim)
            
        elif specialty_type == "context_memory":
            # Focus sur la mémoire conversationnelle
            self.processor = nn.LSTM(embed_dim, embed_dim, batch_first=True, dropout=0.1)
            self.norm = nn.LayerNorm(embed_dim)
            
        elif specialty_type == "social_cues":
            # Focus sur les indices sociaux (politesse, familiarité)
            self.processor = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
            self.activation = nn.ReLU()
            self.norm = nn.LayerNorm(embed_dim)
            
        elif specialty_type == "turn_taking":
            # Focus sur l'alternance de parole
            turn_heads = max(1, embed_dim // 64)
            self.processor = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=turn_heads, dim_feedforward=embed_dim*2,
                dropout=0.1, batch_first=True, activation='gelu'
            )
            self.norm = nn.LayerNorm(embed_dim)
            
        elif specialty_type == "common_sense":
            # Focus sur le sens commun conversationnel
            self.processor = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(0.05),  # Moins de dropout pour préserver le sens commun
                nn.Linear(embed_dim, embed_dim)
            )
            self.norm = nn.LayerNorm(embed_dim)
            
    def forward(self, x):
        if self.specialty_type == "dialog_flow":
            processed, _ = self.processor(x, x, x)
            return self.norm(x + processed)
            
        elif self.specialty_type == "sentiment":
            processed = self.processor(x)
            return self.norm(x + processed)
            
        elif self.specialty_type == "context_memory":
            processed, _ = self.processor(x)
            return self.norm(x + processed)
            
        elif self.specialty_type == "social_cues":
            x_conv = x.transpose(1, 2)
            processed = self.activation(self.processor(x_conv)).transpose(1, 2)
            return self.norm(x + processed)
            
        elif self.specialty_type == "turn_taking":
            processed = self.processor(x)
            return self.norm(x + processed)
            
        elif self.specialty_type == "common_sense":
            processed = self.processor(x)
            return self.norm(x + processed)

class ConversationalBranch(nn.Module):
    """Branche spécialisée pour le dialogue"""
    def __init__(self, embed_dim, num_heads, specialty_type):
        super().__init__()
        self.specialty_type = specialty_type
        
        # Pré-processeur conversationnel
        self.preprocessor = ConversationalPreprocessor(embed_dim, specialty_type)
        
        # Attention multi-têtes
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        
        # Couches de sortie adaptées au dialogue
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 3),  # Plus de capacité pour le langage naturel
            nn.GELU(),  # Activation plus douce
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 3, embed_dim),
            nn.Dropout(0.1)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # 1. Pré-traitement conversationnel
        specialized_input = self.preprocessor(x)
        
        # 2. Attention multi-têtes
        attn_output = self.attention(specialized_input)
        x = self.norm1(specialized_input + attn_output)
        
        # 3. Feed Forward Network
        ffn_output = self.ffn(x)
        output = self.norm2(x + ffn_output)
        
        return output

class SuperNeuronConversational(nn.Module):
    """Super Neuron optimisé pour les conversations"""
    def __init__(self, vocab_size, embed_dim=192, num_heads=6, max_seq_len=64, num_branches=6):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.num_branches = num_branches
        
        # Embeddings adaptés au dialogue
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Spécialisations conversationnelles
        conversation_specialties = [
            "dialog_flow", "sentiment", "context_memory", 
            "social_cues", "turn_taking", "common_sense"
        ]
        
        # Branches spécialisées pour la conversation
        self.branches = nn.ModuleList([
            ConversationalBranch(embed_dim, num_heads, conversation_specialties[i % len(conversation_specialties)])
            for i in range(num_branches)
        ])
        
        # Routage adaptatif pour conversation
        self.router = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_branches),
            nn.Softmax(dim=-1)
        )
        
        # Fusion conversationnelle
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.05)  # Moins agressif pour préserver la fluidité
        )
        
        # Tête de sortie
        self.output_head = nn.Linear(embed_dim, vocab_size)
        
        # Initialisation douce pour conversation
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)  # Plus doux
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(positions)
        x = token_embeds + pos_embeds
        
        # Traitement par branches conversationnelles
        branch_outputs = []
        for branch in self.branches:
            branch_output = branch(x)
            branch_outputs.append(branch_output)
        
        # Routage conversationnel (basé sur le contexte global)
        routing_input = x.mean(dim=1)
        routing_weights = self.router(routing_input)
        
        # Fusion pondérée
        fused_output = torch.zeros_like(branch_outputs[0])
        for i, branch_output in enumerate(branch_outputs):
            weight = routing_weights[:, i].unsqueeze(1).unsqueeze(2)
            fused_output += weight * branch_output
        
        # Fusion finale
        fused_output = self.fusion(fused_output)
        
        # Projection vers vocabulaire
        logits = self.output_head(fused_output)
        
        return logits

class ConversationalLM:
    """Modèle de langage conversationnel complet"""
    def __init__(self):
        self.vocab = {}
        self.vocab_size = 0
        self.model = None
        
    def build_vocab(self, texts):
        """Construction du vocabulaire conversationnel"""
        print("Construction du vocabulaire conversationnel...")
        
        # Tokens spéciaux pour conversation
        special_tokens = ['<pad>', '<unk>', '<start>', '<end>', '<turn>', '<question>', '<response>']
        
        all_words = []
        for text in texts:
            # Nettoyage adapté au langage conversationnel avec toutes les ponctuations
            words = text.lower().replace('!', ' !').replace('?', ' ?').replace('.', ' .').replace(',', ' ,').replace(';', ' ;').replace(':', ' :').replace('-', ' ').split()
            all_words.extend(words)
        
        # Comptage et vocabulaire
        word_counts = Counter(all_words)
        
        # Vocabulaire avec tokens spéciaux + TOUS les mots (même 1 occurrence)
        vocab_list = special_tokens + [word for word, count in word_counts.most_common() if count >= 1]
        
        self.vocab = {word: i for i, word in enumerate(vocab_list)}
        self.vocab_size = len(self.vocab)
        self.reverse_vocab = {i: word for word, i in self.vocab.items()}
        
        print(f"Taille du vocabulaire conversationnel: {self.vocab_size}")
        return self.vocab_size
        
    def encode(self, text):
        """Tokenization conversationnelle"""
        words = text.lower().replace('!', ' !').replace('?', ' ?').replace('.', ' .').replace(',', ' ,').replace(';', ' ;').replace(':', ' :').replace('-', ' ').split()
        return [self.vocab.get(word, self.vocab['<unk>']) for word in words]
    
    def decode(self, tokens):
        """Détokenization avec gestion des tokens inconnus"""
        words = []
        for token in tokens:
            if isinstance(token, torch.Tensor):
                token = token.item()
            if token in self.reverse_vocab:
                word = self.reverse_vocab[token]
                # Filtrer les tokens spéciaux en sortie
                if word not in ['<pad>', '<start>', '<end>']:
                    words.append(word)
            else:
                # Debug: token ID inconnu
                print(f"⚠️  Token ID inconnu: {token} (vocab_size: {self.vocab_size})")
                
        return ' '.join(words).replace(' !', '!').replace(' ?', '?').replace(' .', '.')
    
    def create_model(self, num_branches=6, embed_dim=192, num_heads=6):
        """Création du modèle conversationnel"""
        self.model = SuperNeuronConversational(
            self.vocab_size, embed_dim, num_heads, num_branches=num_branches
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Paramètres totaux: {total_params:,}")
        print(f"Branches conversationnelles: {num_branches}")
        return self.model
    
    def prepare_training_data(self, texts, seq_length=20):
        """Préparation optimisée pour conversation"""
        training_pairs = []
        for text in texts:
            tokens = [self.vocab['<start>']] + self.encode(text) + [self.vocab['<end>']]
            
            # Séquences plus courtes pour dialogue naturel
            for i in range(len(tokens) - seq_length):
                input_seq = tokens[i:i+seq_length]
                target_seq = tokens[i+1:i+seq_length+1]
                training_pairs.append((input_seq, target_seq))
        
        return training_pairs
    
    def generate(self, prompt, max_length=30, temperature=0.9):
        """Génération conversationnelle"""
        if self.model is None:
            raise ValueError("Modèle non initialisé")
            
        self.model.eval()
        tokens = [self.vocab['<start>']] + self.encode(prompt)
        
        with torch.no_grad():
            for _ in range(max_length):
                input_tensor = torch.tensor([tokens[-20:]]).long()  # Contexte limité
                logits = self.model(input_tensor)
                
                # Génération avec température pour naturalité
                next_logits = logits[0, -1, :] / temperature
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                tokens.append(next_token)
                
                if next_token == self.vocab.get('<end>', -1):
                    break
        
        # Retirer tokens spéciaux
        response_tokens = [t for t in tokens if t not in [
            self.vocab.get('<start>', -1), self.vocab.get('<end>', -1)
        ]]
        
        return self.decode(response_tokens)

def main():
    print("=== SUPER NEURON CONVERSATIONNEL ===")
    
    # 1. Chargement du dataset conversationnel
    print("1. Chargement du dataset conversationnel...")
    with open('conversational_dataset.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = data['conversational_texts']
    print(f"Nombre de conversations: {len(texts)}")
    
    # 2. Construction du vocabulaire
    print("2. Construction du vocabulaire...")
    llm = ConversationalLM()
    llm.build_vocab(texts)
    
    # 3. Création du modèle conversationnel
    print("3. Création du modèle conversationnel...")
    model = llm.create_model(num_branches=6, embed_dim=192, num_heads=6)
    
    # 4. Préparation des données
    print("4. Préparation des données conversationnelles...")
    training_pairs = llm.prepare_training_data(texts, seq_length=16)
    print(f"Nombre de paires d'entraînement: {len(training_pairs)}")
    
    # 5. Entraînement conversationnel
    print("5. Entraînement conversationnel...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    model.train()
    
    for epoch in range(10):  # Plus d'epochs pour conversation
        epoch_start = time.time()
        total_loss = 0
        batch_count = 0
        
        with tqdm(training_pairs, desc=f"Epoch {epoch+1}/10", leave=True) as pbar:
            for i, (input_seq, target_seq) in enumerate(pbar):
                
                input_tensor = torch.tensor([input_seq]).long()
                target_tensor = torch.tensor(target_seq).long()
                
                logits = model(input_tensor)
                loss = criterion(logits.squeeze(0), target_tensor)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Plus conservateur
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                if i % 200 == 0:
                    current_loss = total_loss / (batch_count + 1)
                    pbar.set_postfix({'Loss': f'{current_loss:.4f}'})
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(training_pairs)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Temps = {epoch_time:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Temps total: {total_time:.1f}s ({total_time/60:.1f}min)")
    
    # 6. Test conversationnel
    print("\n6. Test de conversation:")
    conversation_starters = [
        "Bonjour",
        "Comment allez-vous",
        "Qu'est-ce que vous",
        "J'aimerais bien",
        "C'est formidable"
    ]
    
    for prompt in conversation_starters:
        response = llm.generate(prompt, max_length=15, temperature=0.8)
        print(f"'{prompt}' → '{response}'")
    
    print("\nModèle conversationnel prêt ! 🗣️")

if __name__ == "__main__":
    main()
