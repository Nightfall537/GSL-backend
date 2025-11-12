# GPU Training Guide for GSL Model

## Current Status

✅ **TensorFlow 2.20.0 installed**
⚠️ **GPU not detected** - Running on CPU

## Why GPU Training?

- **10-50x faster** training compared to CPU
- Can handle larger batch sizes
- Enables mixed precision training
- Better for deep learning models

## Your System

- **OS**: Windows
- **Python**: 3.12.8
- **TensorFlow**: 2.20.0
- **GPU**: NVIDIA (not yet configured)

## How to Enable GPU Training

### Step 1: Check Your NVIDIA GPU

```bash
# Open Command Prompt and run:
nvidia-smi
```

If this works, you have NVIDIA drivers installed. Note your GPU model and CUDA version.

### Step 2: Install CUDA Toolkit

1. Download CUDA Toolkit 11.8 or 12.x from:
   https://developer.nvidia.com/cuda-downloads

2. Choose:
   - OS: Windows
   - Architecture: x86_64
   - Version: Your Windows version
   - Installer Type: exe (local)

3. Run the installer and follow instructions

### Step 3: Install cuDNN

1. Download cuDNN from:
   https://developer.nvidia.com/cudnn
   (Requires NVIDIA Developer account - free)

2. Extract the zip file

3. Copy files to CUDA installation directory:
   ```
   Copy cudnn*/bin/*.dll to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
   Copy cudnn*/include/*.h to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include
   Copy cudnn*/lib/*.lib to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64
   ```

### Step 4: Set Environment Variables

Add to PATH:
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp
```

### Step 5: Verify GPU Detection

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Should show your GPU!

### Step 6: Run GPU Training

```bash
python train_gsl_gpu.py
```

## Alternative: Use CPU Training (Current Setup)

Your current setup works fine on CPU, just slower:

```bash
# Train on CPU (works now)
cd app/models
python train_gsl_improved.py
```

**CPU Training Time**: ~2-4 hours for 100 epochs
**GPU Training Time**: ~10-30 minutes for 100 epochs

## Quick GPU Setup (If You Have NVIDIA Drivers)

If you already have NVIDIA drivers and CUDA installed:

```bash
# Install TensorFlow with GPU support
pip install tensorflow[and-cuda]

# Verify
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Train
python train_gsl_gpu.py
```

## Training Scripts Available

### 1. GPU Training (Requires GPU Setup)
```bash
python train_gsl_gpu.py
```
- Optimized for NVIDIA GPU
- Mixed precision training
- Larger batch sizes (32)
- TensorBoard logging

### 2. CPU Training (Works Now)
```bash
cd app/models
python train_gsl_improved.py
```
- Works on any system
- Smaller batch sizes (8)
- Slower but reliable

### 3. Live Recognition (Works Now)
```bash
python live_sign_recognition.py
```
- Uses trained model
- Real-time webcam recognition
- Prints detected signs

## Performance Comparison

| Hardware | Batch Size | Time/Epoch | Total Time (100 epochs) |
|----------|------------|------------|-------------------------|
| CPU      | 8          | ~2-3 min   | 3-5 hours              |
| GPU      | 32         | ~10-20 sec | 15-30 minutes          |

## Troubleshooting

### "No GPU detected"
- Install NVIDIA drivers
- Install CUDA Toolkit
- Install cuDNN
- Restart computer

### "CUDA out of memory"
- Reduce batch size in training script
- Close other GPU applications
- Use mixed precision training

### "cuDNN not found"
- Verify cuDNN files copied correctly
- Check PATH environment variables
- Restart terminal/IDE

## Recommended Workflow

### For Quick Testing (Use Now):
1. Train on CPU: `cd app/models && python train_gsl_improved.py`
2. Test model: `python live_sign_recognition.py`

### For Production Training (Setup GPU First):
1. Install CUDA + cuDNN
2. Verify GPU: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
3. Train on GPU: `python train_gsl_gpu.py`
4. Deploy model: `python live_sign_recognition.py`

## Current Model Performance

Your existing trained model:
- **15 GSL signs**: BEER, COUSIN, FAMILY, FATHER, FRIEND, FRUIT, GROUP, ME, MINE, MY, NAME, PARENT, RABBIT, RELATIONSHIP, WIFE
- **Validation Accuracy**: ~77-78%
- **Architecture**: Fusion model (MediaPipe + ResNeXt + BiLSTM)
- **Status**: ✅ Ready to use

## Next Steps

1. **Option A - Use Current Model** (Recommended for now)
   ```bash
   python live_sign_recognition.py
   ```

2. **Option B - Train on CPU** (Slower but works)
   ```bash
   cd app/models
   python train_gsl_improved.py
   ```

3. **Option C - Setup GPU** (Best for future)
   - Follow steps above
   - Run `python train_gsl_gpu.py`

## Summary

✅ **Working Now**: Live recognition with existing model
✅ **Working Now**: CPU training (slower)
⏳ **Requires Setup**: GPU training (faster)

Your system is ready for GSL recognition! GPU training is optional but recommended for faster model development.
