# Classification de musiques par genre

Lien du repo : https://gitlab.emi.u-bordeaux.fr/bboidin/m2_son_project/-/tree/master?ref_type=heads

> En équipe avec Aurélie Casanova sur Kaggle.
>
> Durant ce projet, nous avons régulièrement utilisé plusieurs tutoriels et documentations sur les sites suivants :
> StackOverflow, Kaggle, Medium, GitHub.

## Classification par réseau de neurones

Nous avons utilisé un réseau de neurones implémenté avec `keras`:

```python
        model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'), #, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'), #, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(8, activation='softmax')
    ])
```

Nous avons essayé plusieurs architectures, mais celle-ci nous a donné les meilleurs résultats. Nous avons également essayé d'ajouter une régularisation L2, mais cela n'a pas amélioré les résultats.

### Validation

Pour l'entraînement, nous avons fait en sorte de conserver quelques données pour la validation, ce qui nous a permis de détecter le surapprentissage : en effet, si l'accuracy sur les données d'entraînement augmente, et que celle sur les données de validation diminue, c'est que le modèle est en train de surapprendre.

### Overfitting

Les modèles avec plus de couches ont une tendance à overfitter.

Pour résoudre ce problème, nous nous sommes limités à deux couches avec un nombre réduit de neurones, et nous avons ajouté une couche de dropout, qui permet de réduire l'overfitting.

Nous avons également ajouté un `batch_size` de 32, qui permet de plus d'accélérer l'entraînement.

### Activation

Après avoir essayé plusieurs fonctions d'activations telles que sigmoid et tanh, nous avons choisi relu pour les couches cachées et softmax pour la couche de sortie.

### Époques

Augmenter le nombre d'époque n'améliore pas les résultats, et augmente le temps d'entraînement, nous avons donc choisi 10 époques. 

Score Kaggle : 0.67
