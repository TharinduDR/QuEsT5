
import numpy as np

from examples.sentence_level.wmt_2020.common.util.reader import read_annotated_file, read_test_file
from examples.sentence_level.wmt_2020.ro_en.quest5_config import quest5_config
from quest5.algo.run_model import QuEsT5Model

TRAIN_FILE = "examples/sentence_level/wmt_2020/ro_en/data/ro-en/train.roen.df.short.tsv"
DEV_FILE = "examples/sentence_level/wmt_2020/ro_en/data/ro-en/dev.roen.df.short.tsv"
TEST_FILE = "examples/sentence_level/wmt_2020/ro_en/data/ro-en/test20.roen.df.short.tsv"

train = read_annotated_file(TRAIN_FILE)
dev = read_annotated_file(DEV_FILE)
test = read_test_file(TEST_FILE)

train = train[['original', 'translation', 'z_mean']]
dev = dev[['original', 'translation', 'z_mean']]
test = test[['index', 'original', 'translation']]

train["input_text"] = train.apply(lambda x: "sentence1: " + x["original"] + " sentence2: " + x["translation"], axis=1)
dev["input_text"] = dev.apply(lambda x: "sentence1: " + x["original"] + " sentence2: " + x["translation"], axis=1)

train["target_text"] = train["z_mean"].apply(lambda x: np.round(x, decimals=2)).astype(str)
dev["target_text"] = dev["z_mean"].apply(lambda x: np.round(x, decimals=2)).astype(str)

train["prefix"] = "da"
dev["prefix"] = "da"

train = train[["prefix", "input_text", "target_text"]]
dev = dev[["prefix", "input_text", "target_text"]]




model = QuEsT5Model("mt5", "google/mt5-base", args=quest5_config)

model.train_model(train, eval_data=dev)
