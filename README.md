# Statistiques de détections d'oiseaux

*Ce projet est encore une ébauche.*

**Objectif :**

- L'objectif de ce projet est de générer des statistiques de détections d'oiseaux à partir de fichiers audio. On utilise ici BirdNet.

**Ce que ça fait :**

- Prendre une liste de fichiers audio générés typiquement par Merlin Bird Id et générer une liste de détections d'oiseaux ainsi que des figures représentant les détections de la journée.

**Motivation :**

- Merlin Bird Id ne génère pas de statistiques de détections d'oiseaux mais il peut être pertinent d'avoir une liste de détections avec niveaux de confiance lorsque vient le temps de se remémorer quels oiseaux ont été vu lors de la création d'une liste eBird.
- Les oiseaux détectés par birdnetlib sont en anglais. Un fichier CSV `traduction/translation_birds_french_to_english.csv` vient traduire les espèces. S'il en manque lors de la détection, il faut les y ajouter.

**Chronologie du projet :**

- 2024-04-15 : Première version locale pour essayer BirdNet sur un seul fichier audio.
- 2024-04-27 : Seconde version locale avec une boucle pour analyser plusieurs fichiers audio.
- 2024-04-28 : Ajout d'une fonction pour générer des figures.
- 2024-04-28 : Mise sur GitHub.
- 2024-05-10 : Mise à jour fichier de traduction, correction de bugs, simplification du flux de création des figures.
- 2024-05-11 : Correction nom du csv de traduction des noms d'oiseaux en/fr.
- 2024-05-11 : Ajout d'oiseaux dans la liste d'oiseaux traduits.
- 2024-05-11 : Ajout d'une option pour exporter les figures de l'analyse des détections.
- 2024-05-16 : Ajout d'oiseaux dans la liste d'oiseaux traduits.
- 2024-05-16 : Enlevé l'ancien flux et léger nettoyage du code.
- 2024-05-18 : Ajout d'oiseaux dans la liste d'oiseaux traduits.
- 2024-05-18 : Axes du nombre de détections mis en échelle log2 pour faciliter la lecture des graphiques.
- 2024-05-20 : Ajout d'oiseaux dans la liste d'oiseaux traduits.
- 2024-05-20 : Correction d'un bug dans le cas de nom de fichier avec un numéro après l'horodatage.