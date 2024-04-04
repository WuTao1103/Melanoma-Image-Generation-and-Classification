## Automated Melanoma Image Generation and Classification Project

## Project Description
This project aims to automatically generate and classify high-quality melanoma images using the Denoising Diffusion Probabilistic Model (DDPM) and ResNet50 to facilitate early diagnosis and improve treatment outcomes. Through efficient and accurate image processing and analysis techniques, this project enables medical professionals to better understand and diagnose melanoma, and thus provide more personalized and effective treatment plans.

## Dataset Sources
The dataset for this project comes from the International Skin Imaging Collaboration (ISIC), which provides a large collection of expert-reviewed melanoma images to support dermatology research and education.
ISIC:https://challenge.isic-archive.com/data/

### Environment requirements
- Python 3.8+
- PyTorch 1.8+
- torchvision
- denoising_diffusion_pytorch (denoising diffusion modeling library)
- See `requirements.txt` for additional dependencies.

### Installation steps

1. Clone the repository locally:
   ```bash
   git clone https://github.com/WuTao1103/Melanoma-Image-Generation-and-Classification.git

2. Enter the project directory:
   ``bash
   cd Melanoma Image Generation and Classification
   cd Melanoma Image Generation and Classification
3. Install the dependencies (make sure you have pip installed):
   ```bash
   pip install -r requirements.txt
   ```

## DDPM Image Generation
Please refer to README.md in the `ddpm` folder for detailed prediction and training steps.

### Prediction Steps
- Generate images directly using pre-trained weights or use your own training weights.
- The generated images will be saved in the `results/predict_out` directory.

### Training step
- Place the expected image files in the `datasets` folder.
- Run the `train.py` file for training. The images generated during training can be viewed in the `results/train_out` folder.

## ResNet50 image classification
See README.md in the `skin_two_classification` folder for detailed steps.

### Introduction to the code directory
- `args.py`: holds the various parameters used for training and testing.
- `data_gen.py`: implements dataset partitioning, data augmentation and data loading.
- `main.py`: contains training, evaluation and testing.
- `models/Res.py`: rewrites ResNet for various types of networks.

### Run commands
- Training mode: `python main.py --mode=train`.
- Test mode: `python main.py --mode=test --model_path='path/to/your/model.pth'`

## Contribution guidelines
We welcome all forms of contributions, including bug fixes, feature additions, or improvements to the project documentation. Please contact us via issue or pull request.

## License
This project is under the MIT license. See the `LICENSE` file for more details.

## Acknowledgments
We thank the International Skin Imaging Collaboration (ISIC) for providing valuable datasets, as well as all the individuals and organizations that contributed to this project.

