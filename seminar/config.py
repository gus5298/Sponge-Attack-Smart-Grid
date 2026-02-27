import os
import warnings
warnings.filterwarnings("ignore")

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONTEXT_LEN = 64
PREDICTION_LEN = 10
NUM_FEATURES = 9
BATCH_SIZE = 64
MAX_EPOCHS = 50
HIDDEN_SIZE = 512
RNN_LAYERS = 4
LEARNING_RATE = 1e-3

MAX_PONDERS = 20
TIME_PENALTY = 0.01
ACT_HIDDEN_SIZE = 128

DATA_PATH = os.path.join(_BASE_DIR, "data", "Location1.csv")
OUTPUT_DIR = os.path.join(_BASE_DIR, "outputs")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
DEEPAR_MODEL_PATH = os.path.join(MODELS_DIR, "deepar_model.pt")
ACT_MODEL_PATH = os.path.join(MODELS_DIR, "act_model.pt")

POPULATION_SIZE = 20
NUM_PARENTS_MATING = 10
MUTATION_PERCENT = 35
KEEP_ELITISM = 3
REPS_PER_MEASUREMENT = 10
WARMUP_REPS = 5
ENERGY_SAMPLE_INTERVAL = 0.001

BASELINE_REPS = 20
VERIFICATION_REPS = 50
PGD_WARMUP_REPS = 20
PGD_EPSILON = 2.0
PGD_ALPHA = 0.05
PGD_NUM_STEPS = 200

CPU_TDP_WATTS = 65
CPU_IDLE_WATTS = 10

ALL_FEATURES = [
    'Power', 'temperature_2m', 'relativehumidity_2m', 'dewpoint_2m',
    'windspeed_10m', 'windspeed_100m', 'winddirection_10m',
    'winddirection_100m', 'windgusts_10m'
]

FEATURE_NAMES = [
    'Power', 'Temp', 'Humidity', 'DewPoint',
    'Wind10m', 'Wind100m', 'Dir10m', 'Dir100m', 'Gusts'
]
