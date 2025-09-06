# Horse↔Zebra Image Translation Using CycleGAN

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/md-naim-hassan-saykat/skin-cancer-cnn/blob/main/BachelorThesis/skin-cancer.ipynb)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)](https://pytorch.org/get-started/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This project implements **Cycle-Consistent Adversarial Networks (CycleGAN)** in PyTorch for unpaired image-to-image translation between horses and zebras.  
Both directions are trained (Horse→Zebra and Zebra→Horse). The models are evaluated with **qualitative visualizations** and **quantitative metrics (SSIM, PSNR)**.  

---

## Project Overview
- **Framework:** PyTorch  
- **Dataset:** [horse2zebra dataset on Kaggle](https://www.kaggle.com/datasets/suyashdamle/cyclegan) (original CycleGAN paper dataset, mirrored on Kaggle)
- **Generators:** U-Net-like encoder–decoder with residual blocks.  
- **Discriminators:** PatchGAN classifiers.  
- **Losses:**  
  - Adversarial (MSE)  
  - Cycle-consistency (L1)  
- **Optimization:** Adam (lr = 0.0002, betas = (0.5, 0.999))  
- **Training:** 100 epochs, batch size = 1  
- **Evaluation:** SSIM, PSNR  

---

## Repository Structure
horse-to-zebra-cyclegan/
│
├── notebooks/
│   └── cyclegan-horse2zebra.ipynb   # Main Jupyter Notebook
│
├── docs/
│   ├── report.tex                     # LaTeX report
│   ├── references.bib               # References for the report
│   └── report.pdf                     # Compiled project report
│
├── outputs/                         # Saved result figures
│   ├── real_horses.png
│   ├── fake_zebras.png
│   ├── real_zebras.png
│   └── fake_horses.png
│
├── requirements.txt
├── README.md
└── .gitignore

---

## Getting Started

### Clone the repo
git clone [https://github.com/md-naim-hassan-saykat/cyclegan-horse2zebra](https://github.com/md-naim-hassan-saykat/horse-to-zebra-cyclegan).git
cd cyclegan-horse2zebra
## Install dependencies
pip install -r requirements.txt

**Dependencies include:**
- torch, torchvision
- scikit-image (for SSIM, PSNR)
- numpy, tqdm, PIL
- matplotlib
## Download the dataset
# Option 1: Kaggle (recommended)
kaggle datasets download -d suyashdamle/cyclegan -p ./data/
unzip ./data/cyclegan.zip -d ./data/

# Option 2: Direct download (mirror, if Kaggle not available)
wget http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/horse2zebra.zip -O horse2zebra.zip
unzip horse2zebra.zip -d ./data/
## Train the model
Inside the notebook (notebooks/cyclegan-horse2zebra.ipynb), run the training loop:
for epoch in range(num_epochs):
    ...
## Generate results
Run the evaluation cells to produce:
	•	real_horses.png / fake_zebras.png
	•	real_zebras.png / fake_horses.png
## Quantitative Evaluation
	•	SSIM: Structural Similarity Index (10 samples averaged)
	•	PSNR: Peak Signal-to-Noise Ratio (10 samples averaged)

These metrics provide a simple numerical check but are limited for unpaired translation tasks.
## Report
[Read the full project report (PDF)](docs/main.pdf)
## Future Work
	•	Add perceptual metrics: FID, LPIPS
	•	Incorporate semantic/attention-based models to improve structural consistency
	•	Extend to other unpaired datasets
## References
	•	[1] Heusel et al., GANs Trained by a Two Time-Scale Update Rule (FID), NeurIPS 2017.
	•	[2] Isola et al., Image-to-Image Translation with Conditional Adversarial Networks (pix2pix), CVPR 2017.
	•	[3] Wang et al., Image Quality Assessment: SSIM, IEEE TIP 2004.
	•	[4] Zhang et al., Unreasonable Effectiveness of Deep Features (LPIPS), CVPR 2018.
	•	[5] Zhu et al., Unpaired Image-to-Image Translation using CycleGAN, ICCV 2017.

 ## Author

 **Md Naim Hassan Saykat**  
*MSc in Artificial Intelligence, Université Paris-Saclay*  

[LinkedIn](https://www.linkedin.com/in/md-naim-hassan-saykat/)  
[GitHub](https://github.com/md-naim-hassan-saykat)  
[Academic Email](mailto:md-naim-hassan.saykat@universite-paris-saclay.fr)  
[Personal Email](mailto:mdnaimhassansaykat@gmail.com)
