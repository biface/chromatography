# Chromatography

## Présentation

Ce projet vise à modéliser et prédire les chromatogrammes en phase liquide. Il permet de simuler le comportement des
espèces chimiques dans une colonne chromatographique en utilisant des modèles physiques et des méthodes numériques.

## Fonctionnalités

`Chromatography` est une application développée en langage `Rust` qui propose :

- Modélisation des phénomènes d'adsorption en utilisant les isothermes de Langmuir.
- Prédiction des chromatogrammes pour une ou plusieurs espèces chimiques.
- Des méthodes de discrétisation en commençant par la méthode d'Euler, mais des solutions Runge-Kutta ou les différences
  finies sont envisageables.
- Différents profils d'injection disponibles :
    - Impulsion de Dirac (injection instantanée)
    - Profil gaussien (distribution de concentration)
    - Profil rectangulaire (injection continue)
- Interface en ligne de commande pour exécuter les simulations.

