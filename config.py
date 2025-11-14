import os

# ==== Project Paths ====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# ==== Dataset Parameters ====
SAMPLING_RATE = 512        # Hz
EPOCH_SEC = 10             # seconds per EEG segment
N_CHANNELS = 32            # EEG channels
N_CLASSES = 3              # low, medium, high

# ==== Training Parameters ====
BATCH_SIZE = 16
EPOCHS = 50
LR = 0.001
SEED = 42

# ==== Logging and Output Paths ====
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
