
SCALE = 5

WIN_W = 960/SCALE
WIN_H = 400/SCALE

CHAR_W = 50/SCALE
CHAR_H = 80/SCALE

BALL_R = 25/SCALE
BALL_GAP = 20/SCALE

BALL_SPEED = 15/SCALE
BALL_SPAWN_PERIOD = 40 #TIME! not length.

CHAR_SPEED_X = 10/SCALE
CHAR_SPEED_Y = 10/SCALE

# NETWORK PARAMETERS
BATCH_SIZE = 32 # MiniBatch Size
MEM_SIZE = 1e5 # Replay Memory Size
UPDATE_FREQ = 1e3 # Network Update Frequency
TAU = 1./UPDATE_FREQ # For Soft Update, Use Tau instead
LEARNING_RATE = 2e-4 # Learning Rate for Adam Optimizer
EPS_START = 1.0
EPS_END = 0.05
ANNEAL_STEPS = 2e5 # Number of Annealing Steps for Epsilon Decay
PRE_TRAIN_STEPS = 5e4 # Pre-Training to Populate Replay memory
EPS_DELTA = (EPS_END - EPS_START)/ANNEAL_STEPS

GAMMA = 0.99 # Decay Factor for Q-Learning
NUM_EPISODES = 1e4 # Number of Episodes to Train For
TEST_EPISODES = 2e2 # Number of Episodes to Test For
