# CSE274 Project: NeRF

Implementation of Neural Radiance Fields (NeRF) for the CSE274 project.

## Instructions

 - For training: `python3 train.py`
 - For inference: `python3 test.py`

## Repository Structure

 - `train.py`: Contains training code
 - `test.py`: Contains inference code
 - `train_dual.py`: Contains training code for dual network configuration (separate networks for coarse and fine)
 - `test_dual.py`: Contains inference code for dual network configuration
 - `dataloader.py`: Contains the dataloaders
 - `model.py`: Contains the NeRF model
 -  `utils.py`: Contains utility functions for generating rays and volume rendering
 - `params.py`: Contains parameters
 - `checkpoints`: Contains model checkpoints of all the experiments
 - `dataset`: Contains the dataset
 - `losses`: Contains losses of all the experiments
 - `results`: Contains result images of all the experiments
