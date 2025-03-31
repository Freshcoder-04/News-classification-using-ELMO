# News-classification-using-ELMO

## Introduction:

## ELMO
This model contains code to generate bilstm.pt model which stores 3 embeddings for words trained on the brown corpus using input embeddings and a 2 level deep BiLSTM layer ELMO architecture.

```
python3 ELMO.py
```

## Classifiers

### ELMO models
The ``classification.py`` script has definitions for FrozenLambdas, TrainableLambdas and LearnableFunction classes which are used to combine the 3 embeddings achieved from the ELMO model and use these combined embeddings for news-classification task.

```
python3 classification.py
```

### Static Embeddings
The ``static_classification.py`` script generates classifiers on the news-classification dataset using static embeddings from pretrained models ````./models/static-embeddings/svd.pt````, ````./models/static-embeddings/skipgram.pt````, ````./models/static-embeddings/cbow.pt````


## Inference
The script `inference.py` can load models and given a description, tell us the probabilities of that description belonging to each of the 4 classes.

Command to run:
```
python inference.py <saved model path> <description>
```
Output Format:
```
class-1 0.6
class-2 0.2
class-3 0.1
class-4 0.1
```