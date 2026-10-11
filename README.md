# Differential Privacy and Model Accuracy

## Project Overview

This project investigates how the amount and distributional similarity of public data affect the accuracy of differentially private machine learning models. It was developed for the COS 478 final research paper at the University of Maine.

## Experimental Setup

We used DP-SGD with PyTorch and Opacus to classify handwritten digits from the MNIST dataset. The model was pretrained on public data and then privately fine-tuned using DP-SGD.

- Private training samples: 2,000
- Test samples: 1,000
- Training: 20 public epochs and 20 private epochs
- Privacy budget: ε = 8, δ = 10⁻⁵
- Total experiments: 27 runs using 3 random seeds

## Results

**Public Data Amount:** Increasing public samples from 500 to 9,000 improved average accuracy from 84.83% to 91.47%.

**Distributional Closeness:** Rotating public images from 0° to 60° reduced average accuracy from 89.97% to 78.50%.

## Libraries

- [PyTorch](https://pytorch.org/)
- [Opacus](https://opacus.ai/)

Other dependencies are listed in `requirements.txt`.

## Reproducing the Experiments

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the experiments:

```bash
python experiments/run_all_experiments.py
```

The script downloads MNIST automatically and saves the results and graphs in the `results` folder.

## Artifacts

The repository contains the experiment code, CSV results, accuracy graphs, and literature review materials.

## Limitations

MNIST was used to simulate private data. Image rotations were used to introduce controlled distribution shifts rather than measure formal distributional distance. The privacy implementation used non-secure randomness for research experimentation.
