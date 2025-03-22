# 🔍 Reconnaissance d’Empreinte Digitale avec Deep Learning

## 📌 Description
Ce projet implémente un **modèle de Deep Learning** basé sur les **réseaux de neurones convolutifs (CNN)** pour la reconnaissance d’empreintes digitales. Il utilise le dataset **SOCOFing** pour entraîner et évaluer le modèle afin de distinguer les empreintes réelles des empreintes altérées.

## 🎯 Objectifs
✅ Développer un système de reconnaissance d’empreintes digitales 🏆  
✅ Entraîner un modèle CNN pour classifier les empreintes **réelles** et **altérées** 🧠  
✅ Tester et évaluer le modèle sur des images inédites 📊  
✅ Automatiser le processus de reconnaissance biométrique 🔐  

## 🗂 Dataset Utilisé
📂 **SOCOFing** : Un dataset contenant des empreintes digitales **réelles** et **altérées**.  
🔗 [Lien vers le dataset](https://www.kaggle.com/api/v1/datasets/download/ruizgara/socofing)

## 🏗 Architecture du Modèle CNN
Le modèle est basé sur un **réseau de neurones convolutifs (CNN)** et suit ces étapes :
1. **Prétraitement des images** 📷 :
   - Redimensionnement (128x128 pixels)
   - Normalisation des valeurs des pixels
   - Encodage des labels (Real/Altered)
2. **Architecture du CNN** 🧠 :
   - **Couches convolutionnelles** pour extraire les caractéristiques
   - **Couches de pooling** pour réduire la dimensionnalité
   - **Couches entièrement connectées** pour la classification
3. **Entraînement du modèle** 🚀 :
   - Optimisation avec **Adam**
   - Fonction de perte : **Categorical Crossentropy**
   - Évaluation avec **Accuracy Score**
4. **Prédiction** 🔍 :
   - Test du modèle avec de nouvelles empreintes
   - Affichage des résultats avec Matplotlib

## 🔧 Technologies Utilisées
- **Python** 🐍  
- **TensorFlow / Keras** 🧠  
- **OpenCV** (Traitement d’image) 🖼  
- **NumPy, Matplotlib** 📊  
- **PyCharm** (Environnement de développement) 💻  

## 🚀 Utilisation du Code
### 📌 Entraîner le Modèle
Exécuter `main.py` pour entraîner le modèle CNN et sauvegarder le modèle :
```bash
python main.py
```
Le modèle entraîné sera sauvegardé sous **fingerprint_cnn_model.keras**.

### 📌 Tester une Empreinte
Exécuter `predict.py` pour tester une empreinte digitale :
```bash
python predict.py
```

## 📊 Résultats
✅ **Taux de précision élevé** sur le dataset SOCOFing  
✅ **Détection efficace des empreintes altérées**  
✅ **Temps d’inférence rapide** pour une reconnaissance instantanée  
