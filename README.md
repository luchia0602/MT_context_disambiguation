# Improving the quality of machine translation for Japanese with context-based ambiguity resolution
This repository provides code and data for building a context-aware machine translation system focused on resolving contextual ambiguity in Japanese. It includes scripts for data preparation, subtitle alignment and processing, and model training.

Developed as part of a Master’s thesis at the Higher School of Economics.

► preparing-data
  - memory_engine.py: Script generating memory prefixes based on the Japanese input

► processing_data
  - processing subtitles: Folder containing scripts necessary to build a Japanese-English subtitles dataset
  - processing_ami.py: Script building a dataset out of AMI dataset for training a model
  - processing_bpersona.py: Script building a dataset out of BPersona-chat dataset for training a model
  - processing_bsd.py: Script building a dataset out of BSD dataset for training a model
  - processing_east_meld.py: Script building a dataset out of EaST-MELD dataset for training a model
  - processing_manga.py: Script building a dataset out of OpenMantra dataset for training a model

► testing_engine
  - testing_gender.py: Script evaluating a model's ability to detect speakers' genders
  - testing_politeness.py: Script evaluating a model's ability to identify politeness levels

► testing_models
  - document_level_metrics: Scripts for evaluating models' perfomance by computing d-COMET and BERTscore
  - linguistic_phenomena: Scripts for evaluating models' performance in terms of tackling specific linguistic phenomena
  - sentence_level_metrics: Scripts for evaluating models' performance by computing SacreBLEU and chrF++

► training_models
  - active_learning.py: Script for fine-tuning a model using Active Learning
  - al_pool.py: Script for identifying least probable translations out of a samples pool
  - memory_model.py: Main script for training a model with memory prefixes
  - no_memory_model.py: Main script for training a model without memory prefixes

All datasets are available at https://www.kaggle.com/datasets/liudmilashlyakhtina/mem-thesis-en-ja and https://www.kaggle.com/datasets/liudmilashlyakhtina/mem-thesis-ja-en.

All models are available at https://huggingface.co/crowwwwww6.
