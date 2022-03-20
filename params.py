# Parameters
BATCH_SIZE = 1000          # Batch Size
NUM_SAMPLES = 64           # Num of Coarse Samples
NUM_FINE = 128             # Num of Fine Samples
POS_ENCODE_DIMS = 12       # Position encoding L param
VIEWS_ENCODE_DIMS = 4      # Viewing direction encoding L param
EPOCHS = 15000             # Num of Epochs
CHSZ = 256                 # Hidden layers channel size
NUM_LAYERS = 8             # Num of hidden layers
NEAR = 1.0                 # Near distance along ray (Sample lower bound)
FAR = 5.0                  # Far distance along ray (Sample upper bound)
LR = 5e-4                  # Learning Rate
DECAY = 500                # Decay Rate
START = 0                  # Starting Epoch
LATEST = True              # Load from latest.pth if START > 0
DIMENSIONS = 800           # Dimensions of generated image
EXP_NAME = 'theone'        # Name of experiment
