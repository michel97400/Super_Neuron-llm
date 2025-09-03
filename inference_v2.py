#!/usr/bin/env python3
"""
Script d'inférence pour Super Neuron V2 avec pré-processeurs spécialisés
Compatible avec la nouvelle architecture multi-branches
"""

import torch
import torch.nn.functional as F
import os
import sys

# Import du modèle V2
from super_neuron_v2 import SuperNeuronLMV2


def load_trained_model_v2(model_path='saved_models_v2', model_name='super_neuron_v2'):
    """Charge le modèle V2 et tokenizer entraînés"""
    
    print("Chargement du Super Neuron V2...")
    
    try:
        # Créer une instance et charger le modèle
        llm = SuperNeuronLMV2()
        llm.load_model(model_path, model_name)
        
        print(f"✅ Modèle V2 chargé avec {llm.vocab_size} mots dans le vocabulaire")
        print(f"✅ Paramètres du modèle: {sum(p.numel() for p in llm.model.parameters()):,}")
        
        return llm
        
    except FileNotFoundError as e:
        print(f"❌ Erreur: {e}")
        print("Assurez-vous d'avoir d'abord entraîné le modèle avec super_neuron_v2.py")
        return None


def generate_with_analysis(llm, prompt, max_length=15, temperature=1.0, show_routing=False):
    """Génère du texte avec analyse des branches (optionnel)"""
    
    if llm.model is None:
        raise ValueError("Modèle non chargé")
    
    llm.model.eval()
    tokens = llm.encode(prompt)
    
    generated_tokens = []
    routing_history = []
    
    with torch.no_grad():
        for step in range(max_length):
            # Prédiction
            input_tensor = torch.tensor([tokens]).long()
            
            if show_routing:
                # Mode analyse : récupération des poids de routage
                x = llm.model.token_embedding(input_tensor) + llm.model.position_embedding(
                    torch.arange(len(tokens), device=input_tensor.device).unsqueeze(0)
                )
                
                # Calcul des poids de routage
                routing_input = x.mean(dim=1)
                routing_weights = llm.model.router(routing_input)[0].cpu().numpy()
                routing_history.append(routing_weights)
                
                # Prédiction normale
                logits = llm.model(input_tensor)
            else:
                logits = llm.model(input_tensor)
            
            # Sélection du prochain token avec température
            next_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            tokens.append(next_token)
            generated_tokens.append(next_token)
            
            # Arrêt si token de fin
            if next_token == llm.vocab.get('<end>', -1):
                break
    
    generated_text = llm.decode(generated_tokens)
    
    if show_routing:
        return generated_text, routing_history
    else:
        return generated_text


def interactive_generation_v2():
    """Mode interactif pour le Super Neuron V2"""
    print("=== SUPER NEURON V2 - MODE INFÉRENCE ===")
    print("🧠 Architecture: 4 branches spécialisées avec pré-processeurs")
    print("🔧 Branches: Syntax | Semantic | Context | Pattern")
    
    llm = load_trained_model_v2()
    if llm is None:
        return
    
    print("\n" + "="*60)
    print("Commandes disponibles:")
    print("  • Tapez votre prompt pour générer du texte")
    print("  • 'analyze <prompt>' pour voir l'analyse des branches")
    print("  • 'batch' pour tester plusieurs prompts")
    print("  • 'quit' pour quitter")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n🎯 Commande: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Au revoir ! 👋")
                break
                
            if not user_input:
                continue
            
            if user_input.lower().startswith('analyze '):
                # Mode analyse avec routage
                prompt = user_input[8:].strip()
                if not prompt:
                    print("❌ Veuillez spécifier un prompt après 'analyze'")
                    continue
                
                print(f"\n🔍 Analyse du prompt: '{prompt}'")
                result, routing_history = generate_with_analysis(
                    llm, prompt, max_length=10, temperature=1.0, show_routing=True
                )
                
                print(f"📝 Résultat: '{prompt} {result}'")
                print("\n📊 Analyse des branches par étape:")
                
                branch_names = ["Syntax", "Semantic", "Context", "Pattern"]
                for i, weights in enumerate(routing_history[:5]):  # 5 premiers steps
                    print(f"  Step {i+1}:")
                    for j, (name, weight) in enumerate(zip(branch_names, weights)):
                        bar = "█" * int(weight * 20)
                        print(f"    {name:8}: {weight:.3f} {bar}")
                
            elif user_input.lower() == 'batch':
                # Test batch
                batch_generation_v2(llm)
                
            else:
                # Génération normale
                prompt = user_input
                print(f"\n📝 Prompt: '{prompt}'")
                
                # Génération avec différentes températures
                for temp, desc in [(0.7, "🎯 Conservateur"), (1.0, "🔄 Normal"), (1.3, "🚀 Créatif")]:
                    result = generate_with_analysis(llm, prompt, max_length=12, temperature=temp)
                    print(f"{desc} (T={temp}): '{result}'")
                
        except KeyboardInterrupt:
            print("\n\nAu revoir ! 👋")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")


def batch_generation_v2(llm):
    """Test batch pour Super Neuron V2"""
    print("\n🧪 TEST BATCH - SUPER NEURON V2")
    
    test_prompts = [
        "Les algorithmes",
        "L'intelligence artificielle", 
        "Les réseaux de neurones",
        "Le traitement du langage",
        "La programmation",
        "Les données",
        "L'apprentissage automatique",
        "La cybersécurité",
        "Le cloud computing",
        "Les microservices",
        "L'architecture",
        "Les transformers"
    ]
    
    print(f"Test sur {len(test_prompts)} prompts avec analyse de branches:\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        result, routing = generate_with_analysis(
            llm, prompt, max_length=8, temperature=1.0, show_routing=True
        )
        
        # Moyenne des poids de routage
        avg_routing = [sum(step[j] for step in routing) / len(routing) for j in range(4)]
        dominant_branch = ["Syntax", "Semantic", "Context", "Pattern"][avg_routing.index(max(avg_routing))]
        
        print(f"{i:2d}. '{prompt}' → '{result}'")
        print(f"    🧠 Branche dominante: {dominant_branch} ({max(avg_routing):.2f})")


def compare_models():
    """Compare les performances V1 vs V2 (si V1 disponible)"""
    print("\n🆚 COMPARAISON MODÈLES")
    
    # Charger V2
    llm_v2 = load_trained_model_v2()
    if llm_v2 is None:
        return
    
    # Vérifier si V1 existe
    v1_exists = os.path.exists('saved_models/super_neuron_model.pth')
    
    test_prompts = ["Les algorithmes", "L'intelligence artificielle", "Les réseaux de neurones"]
    
    print(f"\nTest comparatif sur {len(test_prompts)} prompts:")
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: '{prompt}'")
        
        # V2 (toujours disponible)
        result_v2 = generate_with_analysis(llm_v2, prompt, max_length=10, temperature=1.0)
        print(f"  V2 (spécialisé): '{result_v2}'")
        
        if v1_exists:
            try:
                # Tenter de charger V1 (nécessiterait l'ancien code)
                print(f"  V1 (classique): [Ancien modèle disponible]")
            except:
                print(f"  V1 (classique): [Non compatible]")
        else:
            print(f"  V1 (classique): [Non entraîné]")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'batch':
            llm = load_trained_model_v2()
            if llm:
                batch_generation_v2(llm)
        elif sys.argv[1] == 'compare':
            compare_models()
        else:
            print("Usage: python inference_v2.py [batch|compare]")
    else:
        interactive_generation_v2()
