# Sentiment Analysis on the IMDB Dataset
Sentiment Analysis aims to classify movie reviews as either positive or negative, leveraging natural language processing and deep learning techniques.

This project demonstrates how to build a Sentiment Analysis model to classify movie reviews from the IMDB dataset as positive or negative using TF Datasets, BERT, TensorFlow and Input Pipelines.

## Overview
This repository contains the code and notebook for performing Sentiment Analysis on the IMDB movie reviews dataset. The model architecture builds on top of the BERT model by fine-tuning it for a sentiment analysis task. The dataset is loaded with TF Datasets, analysed and cleaned. This data is then used to fine-tune a BERT model and tested against the test data. The model’s performance is visualized using accuracy and loss plots.

## Features
- Preprocessing the IMDB dataset to prepare for model training.
- Converting the dataset into BERT-friendly format using TF Datasets.
- Building a custom model with BERT, some dropout, and a sigmoid output layer (for probability estimation).
- Evaluating the model’s performance on test data.
- Visualizing model performance with accuracy and loss curves.
You can find the complete walkthrough and implementation in the `Fine_Tune_BERT_for_Text_Classification_with_TensorFlow.ipynb` notebook provided in this repository.

## References
- Fine-tuning a BERT model - https://www.tensorflow.org/tfmodels/nlp/fine_tune_bert
- BERT, Illustrated for sentence classification - https://jalammar.github.io/illustrated-bert/
- How Transformers outperform RNNs in NLP - https://appinventiv.com/blog/transformer-vs-rnn

## Contributing
Contributions are welcome! Please feel free to submit a pull request if 
you have suggestions for improving the project.

## License
This project is licensed under the MIT License.