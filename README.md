# ToxicBERT-toxicity-classification

Toxicity classification via Fine-tuning BERT

*When first built toxicity models, they found that the models incorrectly learned to associate the names of frequently attacked identities with toxicity. Models predicted a high likelihood of toxicity for comments containing those identities (e.g. "gay"), even when those comments were not actually toxic (such as "I am a gay woman"). This happens because training data was pulled from available sources where unfortunately, certain identities are overwhelmingly referred to in offensive ways. Training a model from data with these imbalances risks simply mirroring those biases back to users.*

This repository shows how to train bert model on : [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)

## Prerequisites:
    + tensorflow==1.13.1
    + numpy==1.16.3
    | Python 3.6 - Anaconda Python Distribution (Recommended)|

## Getting Started with BERT
**Download the pre-trained model**
```
+ [Goolge BERT](https://github.com/google-research/bert)
*Multiple versions of BERT is available. Select as per your needs. I recommend 'uncased_L-12_H-768_A-12' version/*

+ Unzip it and put it in the __data__ folder.

+ Put train, validation and test dataset into __data__ folder.
```

## How to train the mode ?
**It's easy**
```
# To train model
$ ./run_model.sh

# To train and validate model
$ ./run_model_and_val.sh

# To only validate
$ ./run_validate.sh

# To only perform predictions
$ ./run_predict.sh
```


Drop me a mail or connect with me on [Linkedin](https://linkedin.com/in/kumar-nityan-suman/) .

If you like the work I do, show your appreciation by 'FORK', 'START', or 'SHARE'.