"""
Setup GPU Training Environment

This script helps you set up your NVIDIA GPU for TensorFlow training.
"""

import subprocess
import sys
import platform

print("=" * 70)
print("GPU TRAINING SETUP")
print("=" * 70)

print("\n🔍 System Information:")
print(f"  OS: {platform.system()}")
print(f"  Python: {sys.version.split()[0]}")

# Check current TensorFlow
print("\n📦 Checking current TensorFlow installation...")
try:
    import tensorflow as tf
    print(f"  Current TensorFlow: {tf.__version__}")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"  ✓ GPU detected: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"    - {gpu.name}")
        print("\n✅ Your GPU is already configured!")
        sys.exit(0)
    else:
        print("  ⚠️ No GPU detected")
except ImportError:
    print("  TensorFlow not installed")

print("\n" + "=" * 70)
print("GPU SETUP INSTRUCTIONS")
print("=" * 70)

print("""
To enable GPU training on your NVIDIA laptop, follow these steps:

1️⃣ INSTALL NVIDIA DRIVERS
   - Download from: https://www.nvidia.com/Download/index.aspx
   - Or use GeForce Experience to update drivers
   - Restart your computer after installation

2️⃣ INSTALL CUDA TOOLKIT
   - TensorFlow 2.20 supports CUDA 11.8 or 12.x
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Choose your OS and follow installation instructions

3️⃣ INSTALL cuDNN
   - Download from: https://developer.nvidia.com/cudnn
   - Extract and copy files to CUDA installation directory

4️⃣ INSTALL TENSORFLOW WITH GPU SUPPORT
   Run this command:
""")

print("   pip install tensorflow[and-cuda]")

print("""
5️⃣ VERIFY GPU INSTALLATION
   After installation, run:
""")

print("   python -c \"import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))\"")

print("""
=" * 70)

🚀 QUICK START (if you have NVIDIA drivers already):

For Windows with NVIDIA GPU:
""")

response = input("\nWould you like to install TensorFlow with GPU support now? (y/n): ")

if response.lower() == 'y':
    print("\n📦 Installing TensorFlow with GPU support...")
    print("This may take a few minutes...")
    
    try:
        # Uninstall current TensorFlow
        print("\n1. Uninstalling current TensorFlow...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "tensorflow"], 
                      check=False)
        
        # Install GPU version
        print("\n2. Installing TensorFlow with GPU support...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "tensorflow[and-cuda]"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("\n✅ TensorFlow with GPU support installed!")
            print("\n🔍 Verifying GPU detection...")
            
            # Verify
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            
            if gpus:
                print(f"\n🎉 SUCCESS! GPU detected: {len(gpus)} device(s)")
                for gpu in gpus:
                    print(f"  - {gpu.name}")
                print("\n✅ You're ready for GPU training!")
                print("\nRun: python train_gsl_gpu.py")
            else:
                print("\n⚠️ TensorFlow installed but GPU not detected")
                print("\nPossible issues:")
                print("1. NVIDIA drivers not installed")
                print("2. CUDA toolkit not installed")
                print("3. GPU not compatible (requires CUDA Compute Capability 3.5+)")
                print("\nYou can still train on CPU (slower)")
        else:
            print("\n❌ Installation failed")
            print(result.stderr)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease install manually:")
        print("  pip install tensorflow[and-cuda]")
else:
    print("\n💡 When ready, install with:")
    print("   pip install tensorflow[and-cuda]")
    print("\nThen run:")
    print("   python train_gsl_gpu.py")

print("\n" + "=" * 70)
