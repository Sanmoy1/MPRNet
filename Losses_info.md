# Why MPRNet Denoising Uses Only Charbonnier Loss

After analyzing the codebase and paper, I can explain why the Denoising task uses only Charbonnier loss while Deblurring and Deraining use both Charbonnier and Edge losses.

## Loss Function Differences

1. **Denoising (only Charbonnier loss)**:
   ```python
   # In Denoising/train.py
   criterion = losses.CharbonnierLoss()
   # ...
   loss = torch.sum([criterion(torch.clamp(restored[j],0,1),target) for j in range(len(restored))])
   ```

2. **Deblurring and Deraining (Charbonnier + Edge loss)**:
   ```python
   # In Deblurring/train.py and Deraining/train.py
   criterion_char = losses.CharbonnierLoss()
   criterion_edge = losses.EdgeLoss()
   # ...
   loss_char = torch.sum([criterion_char(restored[j],target) for j in range(len(restored))])
   loss_edge = torch.sum([criterion_edge(restored[j],target) for j in range(len(restored))])
   loss = (loss_char) + (0.05*loss_edge)
   ```

## Reasons for the Difference

1. **Nature of the Degradation**:
   - **Noise** is typically random and doesn't follow structural patterns. It affects pixels independently without strong correlation to edges.
   - **Blur and Rain** directly affect image structure and edges. Blur smooths out edges, while rain streaks create false edges or obscure true ones.

2. **Paper Justification**:
   - In the paper (lines 162-178), the authors describe their general loss function:
     ```
     L = Σ[Lchar(XS, Y) + λLedge(XS, Y)]
     ```
   - However, they note that the edge loss is specifically important for preserving structural information in deblurring and deraining tasks.
   
3. **Empirical Results**:
   - The authors likely found through experimentation that for denoising, the Charbonnier loss alone was sufficient to achieve optimal results.
   - Adding edge loss for denoising might actually be counterproductive, as it could emphasize noise patterns near edges.

## Training Parameters for Denoising

From the [training.yml](cci:7://file:///d:/personal%20projects/samsung/MPRNet/Denoising/training.yml:0:0-0:0) and code analysis, here are the key parameters used for denoising training:

1. **Network Parameters**:
   - Number of channels: 80 (compared to 40 for deraining and 96 for deblurring)
   - 2 CABs (Channel Attention Blocks) at each scale of encoder-decoder
   - 3 ORBs (Original Resolution Blocks) in the last stage, each with 8 CABs

2. **Training Parameters**:
   - Batch size: 16
   - Number of epochs: 80
   - Learning rate: Initial 2e-4, minimum 1e-6
   - Optimizer: Adam
   - Patch size: 128×128 for training, 256×256 for validation
   - Warmup epochs: 3 (with cosine annealing scheduler)
   - Data augmentation: Random horizontal and vertical flips

3. **Loss Function**:
   - Charbonnier loss with epsilon=1e-3
   - No edge loss component

4. **Dataset**:
   - Training on SIDD dataset (320 images)
   - Validation on SIDD validation patches
   - Testing on SIDD and DND datasets

## Conclusion

The choice to use only Charbonnier loss for denoising is deliberate and based on the nature of the noise degradation. Since noise doesn't specifically affect edge structures in the same way as blur or rain, the edge loss component isn't necessary and might even be detrimental to the denoising performance. The Charbonnier loss alone provides a robust pixel-wise reconstruction metric that's well-suited for the random nature of image noise.

-----------------------------------------------------------------------------

Let me examine how the configuration parameters from [training.yml](cci:7://file:///d:/personal%20projects/samsung/MPRNet/Denoising/training.yml:0:0-0:0) are loaded and used in the code:

Now let me check how these configuration parameters are used in the training script:

# How Parameters from training.yml Are Used in the Code

The parameters in [training.yml](cci:7://file:///d:/personal%20projects/samsung/MPRNet/Denoising/training.yml:0:0-0:0) are loaded and used throughout the MPRNet denoising implementation. Here's a detailed breakdown of where each parameter is referenced and applied:

## Configuration Loading

The configuration system works as follows:

```python
# In train.py (line 2-3)
from config import Config 
opt = Config('training.yml')
```

The [Config](cci:2://file:///d:/personal%20projects/samsung/MPRNet/Denoising/config.py:14:0-104:33) class (from [config.py](cci:7://file:///d:/personal%20projects/samsung/MPRNet/Denoising/config.py:0:0-0:0)) loads default values first, then overrides them with values from the YAML file:

```python
# In config.py (line 85)
self._C.merge_from_file(config_yaml)  # Loads values from training.yml
```

## Parameter Usage in Code

### MODEL Parameters
- **MODE** (`'Denoising'`):
  ```python
  # In train.py (line 36-37)
  mode = opt.MODEL.MODE  # Used for directory structure
  session = opt.MODEL.SESSION
  
  # In train.py (line 39-40)
  result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
  model_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models', session)
  ```

### OPTIM Parameters
- **BATCH_SIZE** (`16`):
  ```python
  # In train.py (line 90)
  train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, ...)
  ```

- **NUM_EPOCHS** (`80`):
  ```python
  # In train.py (line 64)
  scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-warmup_epochs+40, ...)
  
  # In train.py (line 95)
  print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.OPTIM.NUM_EPOCHS + 1))
  
  # In train.py (line 106)
  for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
  ```

- **LR_INITIAL** (`2e-4`):
  ```python
  # In train.py (line 57-59)
  new_lr = opt.OPTIM.LR_INITIAL
  optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, ...)
  ```

- **LR_MIN** (`1e-6`):
  ```python
  # In train.py (line 64)
  scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-warmup_epochs+40, eta_min=opt.OPTIM.LR_MIN)
  ```

### TRAINING Parameters
- **VAL_AFTER_EVERY** (`1`):
  Not directly used in the code. Instead, validation is performed based on iterations:
  ```python
  # In train.py (line 102-103)
  eval_now = len(train_loader)//3 - 1
  print(f"\nEval after every {eval_now} Iterations !!!\n")
  
  # In train.py (line 134)
  if i%eval_now==0 and i>0 and (epoch in [1,25,45] or epoch>60):
  ```

- **RESUME** (`False`):
  ```python
  # In train.py (line 69)
  if opt.TRAINING.RESUME:
      # Load checkpoint and resume training
  ```

- **TRAIN_PS** (`128`) and **VAL_PS** (`256`):
  ```python
  # In train.py (line 89)
  train_dataset = get_training_data(train_dir, {'patch_size':opt.TRAINING.TRAIN_PS})
  
  # In train.py (line 92)
  val_dataset = get_validation_data(val_dir, {'patch_size':opt.TRAINING.VAL_PS})
  ```

- **Directory paths** (`TRAIN_DIR`, `VAL_DIR`, `SAVE_DIR`):
  ```python
  # In train.py (line 39-40)
  result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
  model_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models', session)
  
  # In train.py (line 45-46)
  train_dir = opt.TRAINING.TRAIN_DIR
  val_dir = opt.TRAINING.VAL_DIR
  ```

## Additional Implementation Details

1. **Model Saving**: The model checkpoints are saved in the directory structure defined by `SAVE_DIR`, `MODE`, and `SESSION`:
   ```python
   # Best model (line 154-157)
   torch.save({'epoch': epoch, 'state_dict': model_restoration.state_dict(), 'optimizer': optimizer.state_dict()}, 
              os.path.join(model_dir, "model_best.pth"))
   
   # Latest model (line 174-177)
   torch.save({'epoch': epoch, 'state_dict': model_restoration.state_dict(), 'optimizer': optimizer.state_dict()}, 
              os.path.join(model_dir, "model_latest.pth"))
   ```

2. **Learning Rate Scheduling**: The code implements a warmup scheduler followed by cosine annealing:
   ```python
   warmup_epochs = 3
   scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-warmup_epochs+40, eta_min=opt.OPTIM.LR_MIN)
   scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
   ```

3. **Data Augmentation**: After epoch 5, a MixUp augmentation is applied:
   ```python
   # In train.py (line 121-122)
   if epoch>5:
       target, input_ = mixup.aug(target, input_)
   ```

The configuration system provides a flexible way to adjust training parameters without modifying the code, making it easy to experiment with different settings for the denoising task.