
from sklearn.model_selection import train_test_split

from examples.sentence_level.wmt_2020.common.util.draw import print_stat
from examples.sentence_level.wmt_2020.common.util.normalizer import fit, un_fit
from examples.sentence_level.wmt_2020.common.util.reader import read_annotated_file, read_test_file
from quest5.algo.model_args import QuEsT5Args
from quest5.algo.run_model import QuEsT5Model

FOLDS = 5
SEED = 777

TRAIN_FILE = "examples/sentence_level/wmt_2020/ro_en/data/ro-en/train.roen.df.short.tsv"
DEV_FILE = "examples/sentence_level/wmt_2020/ro_en/data/ro-en/dev.roen.df.short.tsv"
TEST_FILE = "examples/sentence_level/wmt_2020/ro_en/data/ro-en/test20.roen.df.short.tsv"

train = read_annotated_file(TRAIN_FILE)
dev = read_annotated_file(DEV_FILE)
test = read_test_file(TEST_FILE)

train = fit(train, 'score')
dev = fit(dev, 'score')

train = train[['original', 'translation', 'score']]
dev = dev[['original', 'translation', 'score']]


dev_sentence_pairs = list(map(list, zip(dev['original'].to_list(), dev['translation'].to_list())))

model_args = QuEsT5Args()
model_args.learning_rate = 1e-4
model_args.manual_seed = 777
model_args.num_train_epochs = 5
model_args.best_model_dir = "outputs/best_model"

# for i in range(5):
#     if os.path.exists(model_args.best_model_dir) and os.path.isdir(
#             model_args.best_model_dir):
#         shutil.rmtree(model_args.best_model_dir)

train_df, eval_df = train_test_split(train, test_size=0.1, random_state=model_args.manual_seed)
model = QuEsT5Model("google/mt5-base", args=model_args)
model.train_model(train_df, eval_df)

model = QuEsT5Model(model_args.best_model_dir)
dev_preds = model.predict(dev_sentence_pairs)

dev['predictions'] = dev_preds
dev = un_fit(dev, 'labels')
dev = un_fit(dev, 'predictions')

print_stat(dev, 'labels', 'predictions')