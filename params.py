# Parameters
BATCH_SIZE = 5000          # Batch Size
NUM_SAMPLES = 64           # Num of Coarse Samples
NUM_FINE = 128             # Num of Fine Samples
POS_ENCODE_DIMS = 12       # Position encoding L param
VIEWS_ENCODE_DIMS = 4      # Viewing direction encoding L param
EPOCHS = 2000              # Num of Epochs
CHSZ = 256                 # Hidden layers channel size
NUM_LAYERS = 8             # Num of hidden layers
NEAR = 2.0                 # Near distance along ray (Sample lower bound)
FAR = 6.0                  # Far distance along ray (Sample upper bound)
LR = 5e-4                  # Learning Rate
START = 0                  # Starting Epoch
LATEST = True              # Load from latest.pth if START > 0
DIMENSIONS = 100           # Dimensions of generated image
VIDEO = False              # Use the video dataset to create 360 degree video
