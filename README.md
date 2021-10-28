# Projet Kaggle : Chest X-Ray (Pneumonia)

Ce projet a été réalisé par BERKANE Samir, SOULA Sarah et BOUARROUDJ Lisa.

## Environnement

Créez un environnement de travail grâce à la commande :

 > $ **conda create --name chest_xray**
 
Puis, installer les packages numpy, cv2, tensorflow et keras, grâce à la commande :

 > $ **conda activate chest_xray**
 > $ **conda install nom_du_package**

## Lancement du programme

Pour lancer le programme, placez-vous dans le bon répertoire où se situe le programme python, puis entrez la commande suivante sur votre invité de commandes :

 > $ **python projet.py chest_xray/train/ chest_xray/test/ chest_xray/val/**
 
 avec 
 
 projet.py : le nom du programme python du projet
 
 chest_xray/train/ : le répertoire du jeu de données d'apprentissage
 
 chest_xray/test/ : le répertoire du jeu de données de test
 
 chest_xray/val/ : le répertoire du jeu de données de validation
 
## Sortie du programme

En sortie sur votre invité de commande, vous obtiendrez la formation du modèle sur les données d'apprentissage au cours des différentes epochs. Vous obtiendrez également deux fénêtres avec les courbes de précision et de perte du modèle.

## Aide

Si vous avez besoin d'aide, tapez la commande suivante :

 > $ **python projet.py -h**  
