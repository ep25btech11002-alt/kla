# KLA Semicon 2.0 Hackathon: AI-Based Restoration of Degraded Images for Semiconductor Inspection

## Overview
This project implements a deep learning model for restoring degraded semiconductor inspection images. The model jointly removes speckle noise, Gaussian noise, and performs super-resolution to recover fine details lost due to downsampling.

## Dataset
The dataset consists of paired degraded and ground truth images in .npy format:
- Degraded images: noisy and downsampled (128x128 or 256x256)
- Ground truth images: clean and full resolution (256x256 or 512x512)

The dataset should be organized as follows:
`
data/
  train/
    ground_truth/   # training ground truth .npy files
    degraded/       # training degraded .npy files
  val/
    ground_truth/   # validation ground truth .npy files
    degraded/       # validation degraded .npy files
`

NOTE : The dataset originally given, was split into train (for training) and val (for validation). These folders have also been attached so that the model can train using the correct path. 

## Installation
1. Clone this repository.
2. Create a virtual environment (optional but recommended):
   `
   python -m venv venv
   source venv/bin/activate  # on Windows: venv\\Scripts\\activate
   `
3. Install the required packages:
   `
   pip install -r requirements.txt
   `

## Training
To train the model, run:
`
python train.py
`
This will train the model using the training and validation splits in the data directory. The best model will be saved to best_model.pth.

## Evaluation
To run inference on a test set (or any directory of degraded images), use the provided evaluation script:
`
python evaluate.py --input_dir /path/to/degraded/images --output_dir /path/to/save/restored --model_path /path/to/model.pth
`
The script will load the model, process all .npy files in the input directory, and save the restored images as .npy files in the output directory. It also prints the average inference time per image.

## Model Architecture
The model is a U-Net with residual blocks and transposed convolutions for upsampling. It takes a single-channel degraded image and outputs a single-channel restored image at twice the resolution (since the degradation is 2x downsampling in the provided dataset).

## Results
After training, the model achieves a loss of approximately 0.07 on the validation set (after a few epochs). 

## Notes
- The evaluation script is standalone and does not require any manual edits.
- Ensure that the input directory contains only .npy files (or the script will ignore others).
- The model expects the degraded images to be normalized per image to [0,1] during inference, matching the preprocessing done during training.

## License
This project is for the KLA Semicon 2.0 hackathon.
