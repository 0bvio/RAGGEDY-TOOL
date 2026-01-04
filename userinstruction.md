# RAGGEDY TOOL - User Instructions

It appears you are encountering a `FileNotFoundError: [Errno 2] No such file or directory: 'ninja'` when installing dependencies. This happens because `llama-cpp-python` requires the `ninja` build system to be installed on your **OS level**, not just in your Python environment, when building from source.

### 1. Install System Dependencies
Run the following command in your terminal to install `ninja` and other build essentials:

```bash
sudo apt update
sudo apt install -y ninja-build build-essential graphviz
```

### 2. Install Python Dependencies
Once the system dependencies are installed, try installing the Python requirements again:

```bash
pip install -r requirements.txt
```

### 3. GPU Acceleration (NVIDIA CUDA)
If you have an NVIDIA GPU, you need the **CUDA Toolkit** installed. The error you encountered (`parameter packs not expanded with '...'`) is a known compatibility issue between older CUDA versions and GCC 11.

**For Pop!_OS (Recommended):**
Pop!_OS provides optimized CUDA packages. Run these commands:
```bash
sudo apt update
sudo apt install -y system76-cuda-latest
```

**For Ubuntu / Manual Installation:**
If you are on standard Ubuntu, ensure you have a recent version of CUDA (12.x is recommended):
1. Install CUDA 12: [NVIDIA Download Page](https://developer.nvidia.com/cuda-downloads)
2. Or via apt:
   ```bash
   sudo apt update
   sudo apt install -y nvidia-cuda-toolkit
   ```

**IMPORTANT: Fixing the GCC 11 Compatibility Error**
If you still see the `std_function.h` error during `pip install`, it means your `nvcc` is trying to use a compiler that is too new for it. You can force it to use an older GCC (like GCC 10) if installed:
```bash
sudo apt install -y gcc-10 g++-10
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-10" pip install llama-cpp-python[server] --upgrade --force-reinstall --no-cache-dir
```

**Otherwise, simply ensure you are using the latest CUDA provided by your OS:**
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python[server] --upgrade --force-reinstall --no-cache-dir
```

### 4. Running the Tool
After successful installation:
1. Launch the UI: `python3 main.py ui`
2. Go to the **Models** tab.
3. Download a model (e.g., Llama-3-8B).
4. Click **Start Server**.
5. Switch to the **Chat** tab and start your session.

### Troubleshooting
If the server fails to start even after successful installation:
- **`AttributeError: 'LlamaModel' object has no attribute 'sampler'`**: This is a known bug in certain versions of `llama-cpp-python` (like 0.3.16). We have updated `requirements.txt` to pin version `0.3.1`. Re-install with:
  ```bash
  CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python[server]==0.3.1 --force-reinstall --no-cache-dir
  ```
- **VRAM Issues**: If loading large models (e.g., 20B+), ensure your GPU has enough VRAM. If it crashes, try a smaller model (8B) or decrease the **Context Length** in the **Models** tab.
- **Segmentation Fault (`Py_RunMain` / `libc.so` in logs)**: This usually means the model file (GGUF) is incompatible with your current `llama-cpp-python` version, or your GPU drivers are having issues.
  - Try a different model or a more recent GGUF quantization.
  - Ensure your NVIDIA drivers are up to date.
  - Try running in CPU-only mode by omitting `CMAKE_ARGS="-DGGML_CUDA=on"` during installation to see if it's a GPU-specific issue.
- **Unsupported model architecture**: If the logs say `unknown model architecture: 'qwen3'`, it means the model you downloaded is too new for the installed version of `llama-cpp-python`. 
  - Try a more established model architecture like `llama` (Llama 3) or `mistral`.
  - Or, if you specifically need the new model, try upgrading `llama-cpp-python` (though version `0.3.1` is recommended for stability with most models).
- **Missing `nvcc`**: If `sudo apt install nvidia-cuda-toolkit` doesn't work, you may need to install the full driver and toolkit from NVIDIA's website or use Pop!_OS's specific CUDA packages: `sudo apt install system76-cuda-latest`.
- **Shared library errors**: Ensure your `LD_LIBRARY_PATH` includes the CUDA library directory (e.g., `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`).
- **Check the Logs**: Always check the **Logs** tab in the UI for the specific error message.
- **Port Conflict**: Ensure no other process is using port `8080`.
