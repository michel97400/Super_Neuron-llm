#!/usr/bin/env python3
"""
Super Neuron Conversationnel Amélioré - Version avec plus de données et séquences longues
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

# Importer les classes depuis conversational_model
from conversational_model import SuperNeuronConversational, ConversationalLM

def main():
    print("=== SUPER NEURON CONVERSATIONNEL AMÉLIORÉ ===")
    
    # 1. Chargement du dataset enrichi
    print("1. Chargement du dataset conversationnel enrichi...")
    with open('conversational_dataset_enhanced.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = data['conversational_texts']
    print(f"Nombre de conversations: {len(texts)}")
    print(f"Exemple: '{texts[0][:100]}...'")
    
    # 2. Création du modèle
    print("2. Construction du vocabulaire conversationnel...")
    llm = ConversationalLM()
    llm.build_vocab(texts)
    
    print("3. Création du modèle conversationnel...")
    # Modèle plus grand pour gérer séquences longues et vocabulaire étendu
    model = llm.create_model(num_branches=4, embed_dim=192, num_heads=6)
    
    # 4. Préparation des données avec séquences adaptées
    print("4. Préparation des données conversationnelles...")
    training_pairs = llm.prepare_training_data(texts, seq_length=16)  # Séquences adaptées
    print(f"Nombre de paires d'entraînement: {len(training_pairs)}")
    
    # 5. Entraînement
    print("5. Entraînement conversationnel...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)  # Learning rate plus faible
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    model.train()
    
    for epoch in range(20):  # Plus d'epochs
        epoch_start = time.time()
        total_loss = 0
        batch_count = 0
        
        # Barre de progression pour l'epoch
        with tqdm(training_pairs, desc=f"Epoch {epoch+1}/20", leave=True) as pbar:
            for i, (input_seq, target_seq) in enumerate(pbar):
                # Conversion en tenseurs
                input_tensor = torch.tensor([input_seq]).long()
                target_tensor = torch.tensor(target_seq).long()
                
                # Forward pass
                logits = model(input_tensor)
                loss = criterion(logits.squeeze(0), target_tensor)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Gradient clipping plus strict
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
        
        # Early stopping si loss très faible
        if avg_loss < 0.1:
            print("⚡ Early stopping: Loss suffisamment faible!")
            break
    
    # Temps total d'entraînement
    total_time = time.time() - start_time
    print(f"\n⏱️  Temps total: {total_time:.1f}s ({total_time/60:.1f}min)")
    
    # 6. Sauvegarde (pas encore implémentée dans ConversationalLM)
    print("6. Modèle entraîné prêt (sauvegarde à implémenter)")
    # llm.save_model("saved_models_conv_enhanced", "conversational_enhanced")
    
    # 7. Test de conversation amélioré
    print("7. Test de conversation améliorée:")
    test_prompts = [
        "Bonjour, comment allez-vous",
        "J'aimerais bien savoir",
        "C'est une excellente idée",
        "Vous avez tout à fait raison",
        "D'ailleurs, vous connaissez",
        "Oh oui, j'adore",
        "Moi j'ai un faible pour"
    ]
    
    print("\n📝 Génération avec différentes températures:")
    for prompt in test_prompts[:3]:
        print(f"\n🎯 Prompt: '{prompt}'")
        
        # Température basse (conservateur)
        result_low = llm.generate(prompt, max_length=20, temperature=0.7)
        print(f"  🔹 T=0.7: '{result_low}'")
        
        # Température normale
        result_mid = llm.generate(prompt, max_length=20, temperature=1.0)
        print(f"  🔸 T=1.0: '{result_mid}'")
        
        # Température haute (créatif)
        result_high = llm.generate(prompt, max_length=20, temperature=1.3)
        print(f"  🔥 T=1.3: '{result_high}'")
    
    print("\n🎉 Modèle conversationnel amélioré prêt !")
    print(f"📊 Résumé: {len(texts)} conversations, {len(training_pairs)} paires, vocab={llm.vocab_size}")

if __name__ == "__main__":
    main()
