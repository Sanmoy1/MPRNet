I'll help you set up and run the denoising part of this repository without using conda. Let me first explore the repository structure to understand what we're working with.

Let me check the main README file first to understand the overall project, and then look at the specific Denoising README for detailed instructions.

Now let me check the specific README for the Denoising part:

Let me check the demo.py file to understand how to run the model for inference:

Let me create a plan for setting up and running the denoising model in this repository without using conda.

Now I'll provide you with step-by-step instructions for setting up and running the denoising model in MPRNet without using conda.

# Setting Up and Running MPRNet Denoising Model Without Conda

Based on my analysis of the repository, here's how to set up and run the denoising model without using conda:

# Setting Up and Running MPRNet Denoising Model Without Conda

## 1. Set up Python environment with required dependencies

Instead of using conda, set up a Python virtual environment:

```bash
# Create a virtual environment
python -m venv mprnet_env

# Activate the virtual environment
# On Windows:
mprnet_env\Scripts\activate
```

Install the required dependencies:

```bash
# Install PyTorch 1.1.0 (or a compatible version)
pip install torch==1.1.0 torchvision==0.3.0

# Install other dependencies
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm
```

Install the warmup scheduler:

```bash
cd pytorch-gradual-warmup-lr
python setup.py install
cd ..
```

## 2. Download and place pretrained denoising model

Download the pretrained denoising model from the provided link in the README:
- [Denoising Model](https://drive.google.com/file/d/1LODPt9kYmxwU98g96UrRA0_Eh5HYcsRw/view?usp=sharing)

Place the downloaded model file in the following directory:
```
d:\personal projects\samsung\MPRNet\Denoising\pretrained_models\
```

Make sure the model file is named `model_denoising.pth` as expected by the demo script.

## 3. Prepare test images for inference

Create an input directory for your noisy test images:
```bash
mkdir -p samples/input
```

Place your noisy images in the `samples/input` directory. The script supports `.jpg`, `.JPG`, `.png`, and `.PNG` formats.

## 4. Run inference using demo.py

Run the demo script specifying the Denoising task:

```bash
python demo.py --task Denoising --input_dir ./samples/input/ --result_dir ./samples/output/
```

This will process all images in the input directory and save the denoised results in the output directory.

## 5. Evaluation (Optional)

If you want to evaluate the model on standard datasets:

### For SIDD dataset:
1. Download SIDD Validation Data and Ground Truth from [SIDD Benchmark](https://www.eecs.yorku.ca/~kamel/sidd/benchmark.php)
2. Place them in `./Denoising/Datasets/SIDD/test/`
3. Run:
   ```bash
   cd Denoising
   python test_SIDD.py --save_images
   ```

### For DND dataset:
1. Download DND Benchmark Data from [DND Downloads](https://noise.visinf.tu-darmstadt.de/downloads/)
2. Place it in `./Denoising/Datasets/DND/test/`
3. Run:
   ```bash
   cd Denoising
   python test_DND.py --save_images
   ```

For PSNR/SSIM evaluation, you'll need MATLAB to run:
```
evaluate_SIDD.m
```

## Summary

The denoised images will be saved in the specified output directory. The model handles various image sizes and automatically pads images to be multiples of 8 pixels (as required by the network architecture) before processing.
