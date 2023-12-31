# Classification de musiques par genre

## Extraction de features grâce à librosa

Pour extraire plus de features, nous avons utilisé la bibliothèque `librosa`. Il suffit de stocker `waveform` et `sample_rate` pour caractériser le morceau, puis utiliser les fonctions disponibles.

Nous avons choisi les métriques suivantes :

```python
    features_mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=20)
    feature_tempo = librosa.feature.tempo(y=waveform, sr=sample_rate)
    feature_chroma = librosa.feature.chroma_stft(y=waveform, sr=sample_rate, n_chroma=12)
    feature_spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)
    feature_spectral_contrast = librosa.feature.spectral_contrast(y=waveform, sr=sample_rate)
    feature_zcr = librosa.feature.zero_crossing_rate(y=waveform)
```

Le résultat est composé de deux fichier csv (train et test), qui pourront par la suite être utilsés pour l'entraînement et la classification.

## Classification CatBoost

CatBoost donne de bons résultats, mais il est difficile de dépasser un score de 0.55. Les meilleurs paramètres selon nos recherches sont les suivants :

```python
    model = cbt.CatBoostClassifier(
        iterations=700,
        learning_rate=0.2,
        depth=6,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        random_seed=42,
        logging_level='Silent'
    )
```

Score Kaggle : 0.53
