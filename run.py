import numpy as np
from utils.defines import *
import os
import json

# Load images, labels, predictions
images = []
labels = []
predictions = []

# Define train_idx, valid_idx, unlabel_idx, data split
train_idx = []
valid_idx = []
split = [UNLABELED for i in range(len(images))]
for i in train_idx:
    split[i] = TRAIN
for i in valid_idx:
    split[i] = VALID

# Define a corpus for this task
from processData.Corpus import AttributeCorpus
corpus = AttributeCorpus(datas=datas,attribute_dict="exampleData/corpus_base.json")
corpus.auto_corpus(feature_num=25000, vqa_num=100)
corpus.export_json(file="exampleData/corpus.json")

# Get the attribute of every data
from processData.AttributeSelection import AttributeSelection
AS = AttributeSelection(datas, labels, corpus_path="exampleData/corpus.json",label_names_dict=label_dict)
AS.match_description_to_data()
AS.save_attributes("exampleData/attribute.json")

# Diagnose 
from processData.Diagnose import ModelDiagnose
MD = ModelDiagnose(labels, predictions, split, "exampleData/attribute.json")
MD.detect_failure_by_label()
MD.detect_prediction_correlation()
MD.detect_failure_prediction_correlation()
MD.detect_failure_in_attribute_combination()

# Repair
selected_unlabeled_idx = MD.unlabel_selection()
generation_attribute_dict = MD.date_generation()