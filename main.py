import os
import numpy as np
import tensorflow as tf
Sequential = tf.keras.models.Sequential
Conv2D = tf.keras.layers.Conv2D #C’est une couche clé pour les modèles de reconnaissance d'images.
MaxPooling2D = tf.keras.layers.MaxPooling2D # diminuer la complexité du modèle.
Flatten = tf.keras.layers.Flatten #convertit les caractéristiques multidimensionnelles obtenues après les couches
                                  # de convolution et de pooling en un vecteur plat (1D)
Dense = tf.keras.layers.Dense #Cette couche est souvent utilisée en fin de réseau pour faire des prédictions.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2

# Définir les chemins
data_dir = "E:M2/SOCOFing/SOCOFing"
real_path = os.path.join(data_dir, "Real")
altered_path = os.path.join(data_dir, "Altered")

# Paramètres du modèle
img_width, img_height = 128, 128
batch_size = 64  #64 images seront utilisées pour mettre à jour les poids du modèle à chaque étape d'entraînement.
epochs = 4  # correspond au nombre total de fois que l'ensemble de données d'entraînement complet sera passé
            # à travers le modèle
num_classes = 2

# Fonction pour charger un sous-ensemble des images
def load_images_from_folder(folder, label, limit=None):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if limit is not None and len(images) >= limit:
            break
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (img_width, img_height))
            images.append(img)
            labels.append(label)
    return images, labels

# Charger 5000 images du dossier "Real"
real_images, real_labels = load_images_from_folder(real_path, "Real", limit=5000)  # Limite d'images
#juste pour tester la première fois et pour l'éxécution ne prend pas
# beaucoup de temps

# Charger 7000 images du dossier "Altered" juste pour tester la première fois et pour l'éxécution ne prend pas
# beaucoup de temps
altered_images = []
altered_labels = []
for altered_type in ["Altered-Easy", "Altered-Medium", "Altered-Hard"]:
    path = os.path.join(altered_path, altered_type)
    imgs, lbls = load_images_from_folder(path, "Altered",limit=7000)  # Limite d'images
    altered_images.extend(imgs)
    altered_labels.extend(lbls)

# Préparer les données
all_images = np.array(real_images + altered_images)
all_labels = np.array(real_labels + altered_labels)

# Normalisation des images
all_images = all_images / 255.0
all_images = all_images.reshape(-1, img_width, img_height, 1)

# Encodage des étiquettes
label_encoder = LabelEncoder()
all_labels = label_encoder.fit_transform(all_labels)
all_labels = tf.keras.utils.to_categorical(all_labels, num_classes=num_classes)

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

# Définition d'un modèle CNN simplifié
model = Sequential([
    tf.keras.Input(shape=(img_width, img_height, 1)),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(num_classes, activation='softmax')
])

# Compilation du modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# Évaluation du modèle
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Sauvegarde du modèle
model.save("fingerprint_cnn_model.keras")

