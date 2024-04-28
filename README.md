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

- 2024-04-15 : Première version locale.
- 2024-04-27 : Seconde version locale.
- 2024-04-28 : Mise sur GitHub.

