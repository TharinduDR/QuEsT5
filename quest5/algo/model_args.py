import json
import os
import sys
from dataclasses import asdict, dataclass, field
from multiprocessing import cpu_count


def get_default_process_count():
    process_count = cpu_count() - 2 if cpu_count() > 2 else 1
    if sys.platform == "win32":
        process_count = min(process_count, 61)

    return process_count


def get_special_tokens():
    return ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]


@dataclass
class QuEsT5Args:
    adam_epsilon: float = 1e-8
    best_model_dir: str = "outputs/best_model"
    checkpoint_path: str = "outputs/checkpoints"
    checkpoint_save_steps = 100
    checkpoint_save_total_limit = 2
    evaluation_steps: int = 100
    learning_rate: float = 1e-4
    manual_seed: int = None
    max_grad_norm: float = 1.0
    max_seq_length: int = 256
    n_gpu: int = 1
    not_saved_args: list = field(default_factory=list)
    num_train_epochs: int = 1
    thread_count: int = None
    train_batch_size: int = 8
    weight_decay: float = 0.0

    def update_from_dict(self, new_values):
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                setattr(self, key, value)
        else:
            raise (TypeError(f"{new_values} is not a Python dict."))

    def get_args_for_saving(self):
        args_for_saving = {
            key: value
            for key, value in asdict(self).items()
            if key not in self.not_saved_args
        }
        return args_for_saving

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model_args.json"), "w") as f:
            args_dict = self.get_args_for_saving()
            json.dump(args_dict, f)

    def load(self, input_dir):
        if input_dir:
            model_args_file = os.path.join(input_dir, "model_args.json")
            if os.path.isfile(model_args_file):
                with open(model_args_file, "r") as f:
                    model_args = json.load(f)

                self.update_from_dict(model_args)
