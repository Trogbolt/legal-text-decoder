# --- IO ---
PATH_TRAIN = "/app/output/train.csv"
PATH_VAL   = "/app/output/val.csv"
PATH_TEST  = "/app/output/test.csv"

OUTPUT_DIR = "/app/output"
MODEL_BEST_DIR = "/app/output/model_best"
MODEL_LAST_DIR = "/app/output/model_last"

# --- Model ---
MODEL_NAME = "bert-base-multilingual-cased"
NUM_LABELS = 5
MAX_LENGTH = 256

# --- Training ---
EPOCHS = 8
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 2

# --- Logging ---
LOG_DIR = "/app/log"
