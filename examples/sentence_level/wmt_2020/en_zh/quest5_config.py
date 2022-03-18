from multiprocessing import cpu_count

SEED = 777
TEMP_DIRECTORY = "temp/data"
RESULT_FILE_DEV = "result_dev.tsv"
RESULT_FILE_TEST = "result_test.tsv"
SUBMISSION_FILE = "predictions.txt"
RESULT_IMAGE = "result.jpg"
GOOGLE_DRIVE = False
DRIVE_FILE_ID = None
MODEL_TYPE = "mt5"
MODEL_NAME = "google/mt5-base"

quest5_config = {
    'output_dir': 'temp/outputs/',
    "best_model_dir": "temp/outputs/best_model",
    'cache_dir': 'temp/cache_dir/',

    'fp16': False,
    'fp16_opt_level': 'O1',
    'max_seq_length': 256,
    'train_batch_size': 4,
    'gradient_accumulation_steps': 1,
    'eval_batch_size': 4,
    'num_train_epochs': 5,
    'weight_decay': 0,
    'learning_rate': 2e-5,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.1,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,
    'do_lower_case': False,

    'logging_steps': 300,
    'save_steps': 300,
    "no_cache": False,
    "no_save": False,
    "save_recent_only": True,
    'save_model_every_epoch': False,
    'n_fold': 3,
    'evaluate_during_training': True,
    "evaluate_during_training_silent": False,
    'evaluate_during_training_steps': 300,
    "evaluate_during_training_verbose": True,
    'use_cached_eval_features': False,
    "save_best_model": True,
    'save_eval_checkpoints': False,
    'tensorboard_dir': None,
    "save_optimizer_and_scheduler": True,

    'regression': True,

    'overwrite_output_dir': True,
    'reprocess_input_data': True,

    'process_count': 1,
    'n_gpu': 1,
    'use_multiprocessing': False,
    "multiprocessing_chunksize": 500,
    'silent': False,

    'wandb_project': "quest5-en-zh",
    'wandb_kwargs': {},

    "use_early_stopping": True,
    "early_stopping_patience": 10,
    "early_stopping_delta": 0,
    "early_stopping_metric": "eval_loss",
    "early_stopping_metric_minimize": True,
    "early_stopping_consider_epochs": False,

    "manual_seed": SEED,

    "config": {},
    "local_rank": -1,
    "encoding": None,
}
