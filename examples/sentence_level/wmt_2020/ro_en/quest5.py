import os
import shutil

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from examples.sentence_level.wmt_2020.common.util.draw import draw_scatterplot, print_stat
from examples.sentence_level.wmt_2020.common.util.normalizer import fit, un_fit
from examples.sentence_level.wmt_2020.common.util.reader import read_annotated_file, read_test_file
from quest5.algo.model_args import QuEsT5Args
from quest5.algo.run_model import QuEsT5Model

import math

FOLDS = 5
SEED = 777

TRAIN_FILE = "examples/sentence_level/wmt_2020/ro_en/data/ro-en/train.roen.df.short.tsv"
DEV_FILE = "examples/sentence_level/wmt_2020/ro_en/data/ro-en/dev.roen.df.short.tsv"
TEST_FILE = "examples/sentence_level/wmt_2020/ro_en/data/ro-en/test20.roen.df.short.tsv"

train = read_annotated_file(TRAIN_FILE)
dev = read_annotated_file(DEV_FILE)
test = read_test_file(TEST_FILE)

train = fit(train, 'z_mean')

train = train[['original', 'translation', 'z_mean']]
dev = dev[['original', 'translation', 'z_mean']]
test = test[['index', 'original', 'translation']]

train["input_text"] = train.apply(lambda x: "source: " + x["original"] + " target: " + x["translation"], axis=1)
dev["input_text"] = dev.apply(lambda x: "source: " + x["original"] + " target: " + x["translation"], axis=1)

train["target_text"] = train["z_mean"].apply(lambda x: np.round(x, decimals=2)).astype(str)

train["prefix"] = "da"
dev["prefix"] = "da"

train = train[["prefix", "input_text", "target_text"]]

dev_preds = np.zeros((len(dev), FOLDS))

to_predict = [
    prefix + ": " + str(input_text)
    for prefix, input_text in zip(dev["prefix"].tolist(), dev["input_text"].tolist())
]


for i in range(FOLDS):

    model_args = QuEsT5Args()
    model_args.num_train_epochs = 20
    model_args.no_save = False
    model_args.fp16 = False
    model_args.learning_rate = 1e-4
    model_args.train_batch_size = 8
    model_args.max_length = 4
    model_args.max_seq_length = 256
    model_args.evaluate_generated_text = True
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_steps = int(
        math.floor(len(train) / (model_args.train_batch_size * 3) / 100.0)) * 100
    model_args.evaluate_during_training_verbose = True
    model_args.use_multiprocessing = False
    model_args.use_multiprocessing_for_evaluation = False
    model_args.use_multiprocessed_decoding = False
    model_args.overwrite_output_dir = True
    model_args.save_recent_only = True
    model_args.manual_seed = SEED * i

    model_type = "mt5"
    model_name = "google/mt5-base"

    model_name_prefix = "ro-en"

    model_args.output_dir = os.path.join(model_name_prefix, "outputs")
    model_args.best_model_dir = os.path.join(model_name_prefix, "outputs", "best_model")
    model_args.cache_dir = os.path.join(model_name_prefix, "cache_dir")

    model = QuEsT5Model(model_type, model_name, args=model_args, use_multiprocessing=False, cuda_device=2)
    train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)
    model.train_model(train_data=train_df, eval_data=eval_df)

    model = QuEsT5Model(model_type, model_args.best_model_dir,
                        use_cuda=torch.cuda.is_available(), args=model_args, cuda_device=2)

    preds = model.predict(to_predict)
    dev_preds[:, i] = [float(p) for p in preds]

dev['predictions'] = dev_preds.mean(axis=1)
dev = un_fit(dev, 'predictions')

dev = dev[["original", "translation", "z_mean", "predictions"]]


print_stat(dev, 'z_mean', 'predictions')
