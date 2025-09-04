#!/usr/bin/env python3
"""
Super Neuron Language Model V2 - Avec pré-processeurs spécialisés
Architecture avec branches parallèles et pré-traitement différencié
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

class SpecializedPreprocessor(nn.Module):
    """Pré-processeur spécialisé pour chaque branche"""
    def __init__(self, embed_dim, specialty_type):
        super().__init__()
        self.specialty_type = specialty_type
        self.embed_dim = embed_dim
        
        if specialty_type == "syntax":
            # Focus sur les relations grammaticales locales
            # Adapter le nombre de têtes selon embed_dim
            syntax_heads = max(1, embed_dim // 64)  # Au moins 1 tête, 1 par 64 dims
            self.processor = nn.MultiheadAttention(embed_dim, syntax_heads, batch_first=True)
            self.norm = nn.LayerNorm(embed_dim)
            
        elif specialty_type == "semantic":
            # Focus sur le sens et les concepts globaux
            self.processor = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.ReLU(),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Dropout(0.1)
            )
            self.norm = nn.LayerNorm(embed_dim)
            
        elif specialty_type == "context":
            # Focus sur les dépendances à long terme
            self.processor = nn.LSTM(embed_dim, embed_dim, batch_first=True)
            self.norm = nn.LayerNorm(embed_dim)
            
        elif specialty_type == "pattern":
            # Focus sur les patterns et structures
            self.processor = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
            self.norm = nn.LayerNorm(embed_dim)
            
        elif specialty_type == "logical":
            # Focus sur les relations logiques et causales
            logical_heads = max(1, embed_dim // 64)  # Adapter les têtes
            self.processor = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=logical_heads, dim_feedforward=embed_dim*2,
                dropout=0.1, batch_first=True
            )
            self.norm = nn.LayerNorm(embed_dim)
            
        elif specialty_type == "temporal":
            # Focus sur les dépendances temporelles
            self.processor = nn.GRU(embed_dim, embed_dim, num_layers=2, batch_first=True, dropout=0.1)
            self.norm = nn.LayerNorm(embed_dim)
            
        else:
            # Spécialisation par défaut
            self.processor = nn.Linear(embed_dim, embed_dim)
            self.norm = nn.LayerNorm(embed_dim)
            
    def forward(self, x):
        if self.specialty_type == "syntax":
            # Attention sur les relations syntaxiques
            processed, _ = self.processor(x, x, x)
            return self.norm(x + processed)  # Connexion résiduelle
            
        elif self.specialty_type == "semantic":
            # Transformation sémantique dense
            processed = self.processor(x)
            return self.norm(x + processed)
            
        elif self.specialty_type == "context":
            # LSTM pour capturer le contexte séquentiel
            processed, _ = self.processor(x)
            return self.norm(x + processed)
            
        elif self.specialty_type == "pattern":
            # Convolution pour détecter les patterns
            # x: [batch, seq, embed] -> [batch, embed, seq] pour conv1d
            x_conv = x.transpose(1, 2)
            processed = self.processor(x_conv).transpose(1, 2)
            return self.norm(x + processed)
            
        elif self.specialty_type == "logical":
            # Transformer encoder pour relations logiques
            processed = self.processor(x)
            return self.norm(x + processed)
            
        elif self.specialty_type == "temporal":
            # GRU pour dépendances temporelles
            processed, _ = self.processor(x)
            return self.norm(x + processed)
            
        else:
            # Transformation par défaut
            processed = self.processor(x)
            return self.norm(x + processed)

class SuperNeuronBranch(nn.Module):
    """Une branche du Super Neuron avec son pré-processeur spécialisé"""
    def __init__(self, embed_dim, num_heads, specialty_type):
        super().__init__()
        self.specialty_type = specialty_type
        
        # Pré-processeur spécialisé
        self.preprocessor = SpecializedPreprocessor(embed_dim, specialty_type)
        
        # Attention multi-têtes
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        
        # Couches de sortie
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # 1. Pré-traitement spécialisé
        specialized_input = self.preprocessor(x)
        
        # 2. Attention multi-têtes
        attn_output = self.attention(specialized_input)
        x = self.norm1(specialized_input + attn_output)
        
        # 3. Feed Forward Network
        ffn_output = self.ffn(x)
        output = self.norm2(x + ffn_output)
        
        return output

class SuperNeuronV2(nn.Module):
    """Super Neuron V2 avec pré-processeurs spécialisés"""
    def __init__(self, vocab_size, embed_dim=256, num_heads=6, max_seq_len=512, num_branches=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.num_branches = num_branches
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Types de spécialisations disponibles
        specialty_types = ["syntax", "semantic", "context", "pattern", "logical", "temporal"]
        
        # Branches spécialisées (nombre paramétrable)
        self.branches = nn.ModuleList([
            SuperNeuronBranch(embed_dim, num_heads, specialty_types[i % len(specialty_types)])
            for i in range(num_branches)
        ])
        
        # Mécanisme de routage intelligent adaptatif
        self.router = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_branches),  # Adapté au nombre de branches
            nn.Softmax(dim=-1)
        )
        
        # Couche de fusion
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Tête de sortie
        self.output_head = nn.Linear(embed_dim, vocab_size)
        
        # Initialisation des poids
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(positions)
        x = token_embeds + pos_embeds
        
        # Traitement par les 4 branches spécialisées
        branch_outputs = []
        for branch in self.branches:
            branch_output = branch(x)
            branch_outputs.append(branch_output)
        
        # Calcul des poids de routage (basé sur la moyenne des tokens)
        routing_input = x.mean(dim=1)  # [batch_size, embed_dim]
        routing_weights = self.router(routing_input)  # [batch_size, 4]
        
        # Fusion pondérée des sorties des branches
        fused_output = torch.zeros_like(branch_outputs[0])
        for i, branch_output in enumerate(branch_outputs):
            weight = routing_weights[:, i].unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1]
            fused_output += weight * branch_output
        
        # Couche de fusion finale
        fused_output = self.fusion(fused_output)
        
        # Projection vers le vocabulaire
        logits = self.output_head(fused_output)
        
        return logits

class SuperNeuronLMV2:
    """Modèle de langage complet avec tokenizer"""
    def __init__(self, vocab_size=None):
        self.vocab = {}
        self.vocab_size = vocab_size
        self.model = None
        
    def build_vocab(self, texts):
        """Construction du vocabulaire"""
        print("Construction du vocabulaire...")
        all_words = []
        for text in texts:
            words = text.lower().split()
            all_words.extend(words)
        
        # Comptage et vocabulaire
        word_counts = Counter(all_words)
        vocab_list = ['<pad>', '<unk>', '<start>', '<end>'] + [word for word, _ in word_counts.most_common()]
        
        self.vocab = {word: i for i, word in enumerate(vocab_list)}
        self.vocab_size = len(self.vocab)
        self.reverse_vocab = {i: word for word, i in self.vocab.items()}
        
        print(f"Taille du vocabulaire: {self.vocab_size}")
        return self.vocab_size
        
    def encode(self, text):
        """Tokenization"""
        words = text.lower().split()
        return [self.vocab.get(word, self.vocab['<unk>']) for word in words]
    
    def decode(self, tokens):
        """Détokenization"""
        words = []
        for token in tokens:
            if isinstance(token, torch.Tensor):
                token = token.item()
            if token in self.reverse_vocab:
                words.append(self.reverse_vocab[token])
        return ' '.join(words)
    
    def create_model(self, num_branches=4, embed_dim=256, num_heads=6):
        """Création du modèle avec nombre de branches paramétrable"""
        self.model = SuperNeuronV2(self.vocab_size, embed_dim, num_heads, num_branches=num_branches)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Paramètres totaux: {total_params:,}")
        print(f"Branches: {num_branches} avec spécialisations variables")
        return self.model
    
    def prepare_training_data(self, texts, seq_length=32):
        """Préparation des données d'entraînement"""
        training_pairs = []
        for text in texts:
            tokens = self.encode(text)
            
            for i in range(len(tokens) - seq_length):
                input_seq = tokens[i:i+seq_length]
                target_seq = tokens[i+1:i+seq_length+1]
                training_pairs.append((input_seq, target_seq))
        
        return training_pairs
    
    def generate(self, prompt, max_length=50, temperature=1.0):
        """Génération de texte"""
        if self.model is None:
            raise ValueError("Modèle non initialisé")
            
        self.model.eval()
        tokens = self.encode(prompt)
        
        with torch.no_grad():
            for _ in range(max_length):
                # Prédiction
                input_tensor = torch.tensor([tokens]).long()
                logits = self.model(input_tensor)
                
                # Sélection du prochain token avec température
                next_logits = logits[0, -1, :] / temperature
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                tokens.append(next_token)
                
                # Arrêt si token de fin
                if next_token == self.vocab.get('<end>', -1):
                    break
        
        return self.decode(tokens)
    
    def save_model(self, model_path="saved_models_v2", model_name="super_neuron_v2"):
        """Sauvegarde du modèle et tokenizer"""
        os.makedirs(model_path, exist_ok=True)
        
        # Sauvegarde du modèle
        model_file = os.path.join(model_path, f"{model_name}_model.pth")
        torch.save(self.model.state_dict(), model_file)
        print(f"Modèle sauvegardé: {model_file}")
        
        # Sauvegarde du tokenizer
        tokenizer_file = os.path.join(model_path, f"{model_name}_tokenizer.pkl")
        tokenizer_data = {
            'vocab': self.vocab,
            'vocab_size': self.vocab_size,
            'reverse_vocab': self.reverse_vocab
        }
        with open(tokenizer_file, 'wb') as f:
            pickle.dump(tokenizer_data, f)
        print(f"Tokenizer sauvegardé: {tokenizer_file}")
        
        return model_file, tokenizer_file
    
    def load_model(self, model_path="saved_models_v2", model_name="super_neuron_v2", num_branches=4):
        """Chargement du modèle et tokenizer"""
        # Chargement du tokenizer
        tokenizer_file = os.path.join(model_path, f"{model_name}_tokenizer.pkl")
        with open(tokenizer_file, 'rb') as f:
            tokenizer_data = pickle.load(f)
        
        self.vocab = tokenizer_data['vocab']
        self.vocab_size = tokenizer_data['vocab_size']
        self.reverse_vocab = tokenizer_data['reverse_vocab']
        
        # Création et chargement du modèle avec branches paramétrables
        self.model = SuperNeuronV2(self.vocab_size, num_branches=num_branches)
        model_file = os.path.join(model_path, f"{model_name}_model.pth")
        self.model.load_state_dict(torch.load(model_file, map_location='cpu'))
        
        print(f"Modèle chargé: {model_file}")
        print(f"Tokenizer chargé: {tokenizer_file}")
        print(f"Configuration: {num_branches} branches")
        return self.model

def main():
    print("=== SUPER NEURON V2 - AVEC PRÉ-PROCESSEURS SPÉCIALISÉS ===")
    
    # 1. Chargement des données
    print("1. Chargement des données...")
    with open('extended_dataset.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = data['training_texts']  # Correction de la clé
    print(f"Nombre de textes: {len(texts)}")
    
    # 2. Création du modèle
    print("2. Construction du vocabulaire...")
    llm = SuperNeuronLMV2()
    llm.build_vocab(texts)
    
    print("3. Création du modèle...")
    model = llm.create_model(num_branches=6, embed_dim=128, num_heads=8)  # 128÷8=16 ✅

    # 4. Préparation des données
    print("4. Préparation des données...")
    training_pairs = llm.prepare_training_data(texts, seq_length=16)
    print(f"Nombre de paires d'entraînement: {len(training_pairs)}")
    
    # 5. Entraînement
    print("5. Entraînement...")
    print(f"Dataset: {len(training_pairs)} paires d'entraînement")
    print(f"Estimation: ~{len(training_pairs) * 5 / 1000:.1f}k itérations pour 5 epochs")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    model.train()

    for epoch in range(5):  # Test rapide avec 3 epochs
        epoch_start = time.time()
        total_loss = 0
        batch_count = 0
        
        # Barre de progression pour l'epoch
        with tqdm(training_pairs, desc=f"Epoch {epoch+1}/5", leave=True) as pbar:
            for i, (input_seq, target_seq) in enumerate(pbar):
                # Entraînement sur TOUT le dataset, pas seulement 100 exemples
                if batch_count >= len(training_pairs):  # Utiliser tout le dataset
                    break
                    
                # Conversion en tenseurs
                input_tensor = torch.tensor([input_seq]).long()
                target_tensor = torch.tensor(target_seq).long()
                
                # Forward pass
                logits = model(input_tensor)
                loss = criterion(logits.squeeze(0), target_tensor)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                # Mise à jour de la barre de progression
                if i % 100 == 0:
                    current_loss = total_loss / (batch_count + 1)
                    pbar.set_postfix({
                        'Loss': f'{current_loss:.4f}',
                        'Batch': f'{batch_count}/{len(training_pairs)}'
                    })
        
        # Statistiques de l'epoch
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(training_pairs)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Temps = {epoch_time:.1f}s")
    
    # Temps total d'entraînement
    total_time = time.time() - start_time
    print(f"\n⏱️  Temps total d'entraînement: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"📊 Débit: {len(training_pairs) * 5 / total_time:.1f} exemples/seconde")
    
    # 6. Sauvegarde
    print("6. Sauvegarde...")
    llm.save_model()
    
    # 7. Test de génération
    print("7. Test de génération:")
    test_prompts = [
        "Les algorithmes",
        "L'intelligence artificielle", 
        "Les réseaux de neurones",
        "Le traitement du langage"
    ]
    
    for prompt in test_prompts:
        generated = llm.generate(prompt, max_length=20, temperature=0.8)
        print(f"'{prompt}' -> '{generated}'")
    
    print("\nEntraînement terminé ! Modèle V2 avec pré-processeurs spécialisés prêt.")

if __name__ == "__main__":
    main()
