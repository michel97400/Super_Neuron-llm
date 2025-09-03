# Super Neuron Language Model 🧠

Un modèle de langage innovant basé sur une architecture "Super Neuron" avec branches multiples et routage intelligent.

## 🚀 Fonctionnalités

- **Architecture Super Neuron** : Modèle avec multiples branches d'attention parallèles
- **Routage intelligent** : Système qui apprend à diriger les tokens vers les bonnes branches
- **Multi-Head Attention** : Implémentation complète du mécanisme d'attention
- **Pipeline complet** : Tokenisation, entraînement, inférence et génération de texte
- **Dataset étendu** : 140 textes sur l'IA, programmation et technologies

## 🏗️ Architecture

```
Input → Embeddings → Super Neuron → Output
                        ↓
        ┌─── Branch 1 (MultiHead) ───┐
        ├─── Branch 2 (MultiHead) ───┤ → Router → Combiner
        ├─── Branch 3 (MultiHead) ───┤
        └─── Branch 4 (MultiHead) ───┘
```

## 📁 Structure du projet

```
.
├── super_neuron_stable.py     # Modèle principal stable
├── extended_dataset.json     # Dataset d'entraînement (140 textes)
├── dataset.json             # Dataset original
└── README.md               # Ce fichier
```

## 🛠️ Installation

```bash
git clone https://github.com/[VOTRE_USERNAME]/super-neuron-llm.git
cd super-neuron-llm
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou .venv\Scripts\activate  # Windows
pip install torch numpy
```

## 🚀 Utilisation

### Entraînement

```bash
python super_neuron_stable.py
```

### Génération de texte

```python
from super_neuron_stable import SuperNeuronLM, SimpleTokenizer, generate_text

# Charger le modèle entraîné
model = SuperNeuronLM(vocab_size=418, d_model=96, num_branches=4)
tokenizer = SimpleTokenizer()

# Générer du texte
prompt = "Les algorithmes"
generated = generate_text(model, tokenizer, prompt)
print(f"'{prompt}' -> '{generated}'")
```

## 📊 Résultats

Le modèle génère du texte cohérent dans le domaine technologique :

- **"Les algorithmes"** → *"deep sciences"*
- **"L'intelligence artificielle"** → *"représente les"*
- **"Les réseaux de neurones"** → *"artificiels le"*
- **"Le traitement du langage"** → *"naturel les"*

## 🎯 Caractéristiques techniques

- **242,630 paramètres** au total
- **4 branches** avec 6 têtes d'attention chacune
- **Vocabulaire** : 418 mots spécialisés
- **Dataset** : 140 textes, 3,220 paires d'entraînement
- **Loss finale** : 0.31 (excellente convergence)

## 🔬 Innovation

Le "Super Neuron" introduit un concept novateur :
- **Routage adaptatif** : Différent pendant l'entraînement et l'inférence
- **Spécialisation des branches** : Chaque branche peut apprendre des patterns différents
- **Robustesse** : Si une branche échoue, les autres compensent

## 🤝 Contribution

Les contributions sont bienvenues ! N'hésitez pas à :
- Signaler des bugs
- Proposer des améliorations
- Ajouter de nouvelles fonctionnalités
- Enrichir le dataset

## 📄 Licence

MIT License - Voir le fichier LICENSE pour plus de détails.

## 🙏 Remerciements

Inspiré par les architectures Transformer et les systèmes Mixture of Experts.

---

**Développé avec ❤️ en Python & PyTorch**
