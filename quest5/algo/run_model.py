import math
import os
import random
import numpy as np
import torch
from sklearn.metrics.pairwise import paired_cosine_distances
from torch.utils.data import DataLoader

from quest5.algo.model_args import QuEsT5Args
from quest5.sentence_transformers import SentenceTransformer, InputExample
from quest5.sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from quest5.sentence_transformers.losses import ContrastiveLoss
from sentence_transformers.models import Transformer, Pooling


class QuEsT5Model:
    def __init__(
            self,
            model_name,
            args=None,
            use_cuda=True,
            cuda_device=-1,
            **kwargs,
    ):

        """
        Initializes a QuEsT5Model model.

        Args:
            model_type: The type of model (t5, mt5, byt5)
            model_name: The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """

        # self.model = SentenceTransformer(model_name)
        self.args = self.load_model_args(model_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, QuEsT5Args):
            self.args = args

        transformer_model = Transformer(model_name, max_seq_length=self.args.max_seq_length)
        pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True,
                                pooling_mode_cls_token=False,
                                pooling_mode_max_tokens=False)
        modules = [transformer_model, pooling_model]
        self.model = SentenceTransformer(modules=modules)

        if self.args.thread_count:
            torch.set_num_threads(self.args.thread_count)

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

    def train_model(self, train_df, eval_df):

        train_samples = []
        for index, row in train_df.iterrows():
            score = float(row["score"])
            inp_example = InputExample(texts=[row['original'], row['translation']], label=score)
            train_samples.append(inp_example)

        eval_samples = []
        for index, row in eval_df.iterrows():
            score = float(row["score"])
            inp_example = InputExample(texts=[row['original'], row['translation']], label=score)
            eval_samples.append(inp_example)

        train_dataloader = DataLoader(train_samples, shuffle=True,
                                      batch_size=self.args.train_batch_size)
        train_loss = ContrastiveLoss(model=self.model)

        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(eval_samples, name='eval')
        warmup_steps = math.ceil(len(train_dataloader) * self.args.num_train_epochs * 0.1)

        self.model.fit(train_objectives=[(train_dataloader, train_loss)],
                       evaluator=evaluator,
                       epochs=self.args.num_train_epochs,
                       evaluation_steps=self.args.evaluation_steps,
                       optimizer_params={'lr': self.args.learning_rate,
                                         'eps': self.args.adam_epsilon},
                       warmup_steps=warmup_steps,
                       weight_decay=self.args.weight_decay,
                       max_grad_norm=self.args.max_grad_norm,
                       output_path=self.args.best_model_dir,
                       checkpoint_save_steps=self.args.checkpoint_save_steps,
                       checkpoint_save_total_limit=self.args.checkpoint_save_total_limit)

        self.save_model_args(self.args.best_model_dir)

    def predict(self, to_predict, verbose=True):
        sentences1 = []
        sentences2 = []

        for text_1, text_2 in to_predict:
            sentences1.append(text_1)
            sentences2.append(text_2)

        embeddings1 = self.model.encode(sentences1, show_progress_bar=verbose, convert_to_numpy=True)
        embeddings2 = self.model.encode(sentences2, show_progress_bar=verbose, convert_to_numpy=True)

        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))

        if len(cosine_scores) == 1:
            return cosine_scores[0]

        else:
            return cosine_scores

    def save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def load_model_args(self, input_dir):
        args = QuEsT5Args()
        args.load(input_dir)
        return args
