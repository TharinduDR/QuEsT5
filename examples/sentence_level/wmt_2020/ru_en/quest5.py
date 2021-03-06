import os
import shutil

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from examples.sentence_level.wmt_2020.common.util.draw import draw_scatterplot, print_stat
from examples.sentence_level.wmt_2020.common.util.normalizer import fit, un_fit
from examples.sentence_level.wmt_2020.common.util.reader import read_annotated_file, read_test_file
from examples.sentence_level.wmt_2020.ru_en.quest5_config import quest5_config, MODEL_TYPE, MODEL_NAME, SEED, \
    TEMP_DIRECTORY, RESULT_FILE_DEV, RESULT_FILE_TEST, RESULT_IMAGE
from quest5.algo.run_model import QuEsT5Model

if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

TRAIN_FILE = "examples/sentence_level/wmt_2020/ru_en/data/ru-en/train.ruen.df.short.tsv"
DEV_FILE = "examples/sentence_level/wmt_2020/ru_en/data/ru-en/dev.ruen.df.short.tsv"
TEST_FILE = "examples/sentence_level/wmt_2020/ru_en/data/ru-en/test20.ruen.df.short.tsv"

train = read_annotated_file(TRAIN_FILE)
dev = read_annotated_file(DEV_FILE)
test = read_test_file(TEST_FILE)

train = fit(train, 'z_mean')

train = train[['original', 'translation', 'z_mean']]
dev = dev[['original', 'translation', 'z_mean']]
test = test[['index', 'original', 'translation']]

train["input_text"] = train.apply(lambda x: "sentence1: " + x["original"] + " sentence2: " + x["translation"], axis=1)
dev["input_text"] = dev.apply(lambda x: "sentence1: " + x["original"] + " sentence2: " + x["translation"], axis=1)
test["input_text"] = test.apply(lambda x: "sentence1: " + x["original"] + " sentence2: " + x["translation"], axis=1)

train["target_text"] = train["z_mean"].apply(lambda x: np.round(x, decimals=1)).astype(str)
# dev["target_text"] = dev["z_mean"].apply(lambda x: np.round(x, decimals=2)).astype(str)

train["prefix"] = "da"
dev["prefix"] = "da"
test["prefix"] = "da"

train = train[["prefix", "input_text", "target_text"]]
# dev = dev[["prefix", "input_text", "target_text"]]

dev_preds = np.zeros((len(dev), quest5_config["n_fold"]))
test_preds = np.zeros((len(test), quest5_config["n_fold"]))

to_predict_dev = [
    prefix + ": " + str(input_text)
    for prefix, input_text in zip(dev["prefix"].tolist(), dev["input_text"].tolist())
]

to_predict_test = [
    prefix + ": " + str(input_text)
    for prefix, input_text in zip(test["prefix"].tolist(), test["input_text"].tolist())
]

for i in range(quest5_config["n_fold"]):
    if os.path.exists(quest5_config['output_dir']) and os.path.isdir(
            quest5_config['output_dir']):
        shutil.rmtree(quest5_config['output_dir'])
    model = QuEsT5Model(MODEL_TYPE, MODEL_NAME, args=quest5_config, use_multiprocessing=False)
    train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)
    model.train_model(train_data=train_df, eval_data=eval_df)

    model = QuEsT5Model(MODEL_TYPE, quest5_config["best_model_dir"],
                        use_cuda=torch.cuda.is_available(), args=quest5_config)

    dev_preds = model.predict(to_predict_dev)
    test_preds = model.predict(to_predict_test)

    dev_preds[:, i] = [float(p) for p in dev_preds]
    test_preds[:, i] = [float(p) for p in test_preds]

dev['predictions'] = dev_preds.mean(axis=1)
dev = un_fit(dev, 'predictions')

test['predictions'] = test_preds.mean(axis=1)
test = un_fit(test, 'predictions')

dev = dev[["original", "translation", "z_mean", "predictions"]]
test = test[["original", "translation", "z_mean", "predictions"]]


dev.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE_DEV), header=True, sep='\t', index=False, encoding='utf-8')
draw_scatterplot(dev, 'z_mean', 'predictions', os.path.join(TEMP_DIRECTORY, RESULT_IMAGE), "Russian-English-Dev")
print_stat(dev, 'z_mean', 'predictions')

test.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE_TEST), header=True, sep='\t', index=False, encoding='utf-8')
draw_scatterplot(test, 'z_mean', 'predictions', os.path.join(TEMP_DIRECTORY, RESULT_IMAGE), "Russian-English-Test")
print_stat(test, 'z_mean', 'predictions')
