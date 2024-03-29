# ToxicBERT [![Codacy Badge](https://api.codacy.com/project/badge/Grade/189b57f845f944868d5219c0094fd719)](https://www.codacy.com/app/nityansuman/toxic-bert?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=nityansuman/toxic-bert&amp;utm_campaign=Badge_Grade)
**Fine-tuning BERT for Toxicity classification**

*When first built toxicity models, they found that the models incorrectly learned to associate the names of frequently attacked identities with toxicity. Models predicted a high likelihood of toxicity for comments containing those identities (e.g. "gay"), even when those comments were not actually toxic (such as "I am a gay woman"). This happens because training data was pulled from available sources where unfortunately, certain identities are overwhelmingly referred to in offensive ways. Training a model from data with these imbalances risks simply mirroring those biases back to users.*

This repository shows how to train bert model on : [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)
- - - -

## Prerequisites:
+ tensorflow==1.13.1
+ numpy==1.16.3
+ Python==3.7 - Anaconda Python Distribution (Recommended)


## Getting Started with Toxic-BERT
**Download the pre-trained model**
+ [Download Goolge BERT Model](https://github.com/google-research/bert) - I recommend 'uncased_L-12_H-768_A-12' version**
+ Unzip and put it in the `__model__` folder.
+ [Download Toxic Dataset](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data/)
+ Put train, dev (if any) and test dataset into `__data__` folder.

```
CSV Data Format:
    Columns: (index, label, comment_text) # Column names are not important. Order is.
```
- - - -

## How to work with this?
**It's easy**
+ Set model checkpoint at --init_checkpoint (bert_model.ckpt file)

*Note: Other than training Checkpoint name available in a file called checkpoint in model_dir folder.*
+ Set pre-trained model weights at ----vocab_file (vocab.txt file) and --bert_config_file (bert_config.json file)

**Train model**
```
$ ./run_model.sh
```
**Train and Validate model: Latest model checkpoint is used for evaluation**
```
$ ./run_model_new.sh
```

**Validate model: Set the model checkpoint to be used at --init_checkpoint**
```
$ ./run_eval.sh
```

**Get predictions: Set the model checkpoint to be used at --init_checkpoint**
```
$ ./run_predict.sh
```
- - - -

Drop me a mail or connect with me on [Linkedin](https://linkedin.com/in/kumar-nityan-suman/) .
If you like the work I do, show your appreciation by 'FORK', 'START', or 'SHARE'.
