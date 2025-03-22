import cv2
import numpy as np
import tensorflow as tf
import os
from matplotlib import pyplot as plt

load_model = tf.keras.models.load_model

# Paramètres
img_width, img_height = 128, 128  # hadi la taille kima utilisinaha f l'entraînement

# NChargiw le modèle ta3na
model = load_model("fingerprint_cnn_model.keras")


# Prétraitement de l'image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_width, img_height))
    img = img / 255.0#Cette ligne divise chaque pixel de l'image par 255.0, ce qui permet de normaliser
                     # les valeurs des pixels entre 0(noir) et 1(blanc).
    img = img.reshape(1, img_width, img_height, 1)#Le dernier 1 : Comme l'image est en niveaux de gris,
                                                  # elle n'a qu'un seul canal.
    return img


# Fonction d'affichage de l'image et du résultat
def show_result(image_path, result_text):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap='gray')
    plt.title(result_text)
    plt.axis('on')
    plt.show()


# Exemple d'utilisation bach ntestou le modèle yla rah yakhdem bien
new_image_path = "E:/M2/f/314__M_Right_ring_finger_Zcut.BMP"

# Vérifier si l'image existe
if os.path.exists(new_image_path):
    print("L'empreinte est trouvée.")
    processed_image = preprocess_image(new_image_path)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    #print("Classe prédite :", "Altered" if predicted_class == 0 else "Real")
    if predicted_class == 1:
        print("Classe prédite :", "Real")
    else:
        print("Classe prédite :", "Altered")

    # Message personnalisé
    if predicted_class == 1:
        result_text = "Information confirm! Fingerprint matches: person exist (Real)"
    else:
        result_text = "Information confirm! Fingerprint matches: person exist (Altered)"

    print(result_text)
    show_result(new_image_path, result_text)
else:
    print("L'empreinte n'existe pas. Vérifiez le chemin du fichier.")
