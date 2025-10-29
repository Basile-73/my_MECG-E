[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mecg-e-mamba-based-ecg-enhancer-for-baseline/ecg-denoising-on-qt-nstdb)](https://paperswithcode.com/sota/ecg-denoising-on-qt-nstdb?p=mecg-e-mamba-based-ecg-enhancer-for-baseline)

# MECG-E: Mamba-based ECG Enhancer for Baseline Wander Removal

### Abstract
Electrocardiogram (ECG) is an important non-invasive method for diagnosing cardiovascular disease. However, ECG signals are susceptible to noise contamination, such as electrical interference or signal wandering, which reduces diagnostic accuracy. Various ECG denoising methods have been proposed, but most existing methods yield suboptimal performance under very noisy conditions or require several steps during inference, leading to latency during online processing. In this paper, we propose a novel ECG denoising model, namely Mamba-based ECG Enhancer (MECG-E), which leverages the Mamba architecture known for its fast inference and outstanding nonlinear mapping capabilities. Experimental results indicate that MECG-E surpasses several well-known existing models across multiple metrics under different noise conditions. Additionally, MECG-E requires less inference time than state-of-the-art diffusion-based ECG denoisers, demonstrating the model's functionality and efficiency. [[paper](https://arxiv.org/abs/2409.18828)]

<p align="center">
<img src="figs/architecture.png"/>
</p>

## Pre-requisites
1. Clone this repository.
2. Install python requirements. Please refer to [Installation](#installation).
3. Download and extract the [ECG data](https://drive.google.com/file/d/19qOwywAoxreEv4xONTk-smQdo-ZdoPBc/view?usp=sharing). Clean ECG records from the [QT Database](https://ieeexplore.ieee.org/document/648140) were corrupted using noise profiles from the MIT-BIH Noise Stress Test Database ([NSTDB](https://physionet.org/content/nstdb/1.0.0/)). Dataset preprocessing follows the procedure outlined in [DeepFilter](https://github.com/fperdigon/DeepFilter/tree/master/Data_Preparation).

## Installation

#### Requirement
    * Python >= 3.9
    * CUDA >= 12.0
    * PyTorch == 2.2.2

#### Environment installation
1. **Create a Python environment with Conda:** It is strongly recommended to set up a dedicated Python environment to manage dependencies effectively and prevent conflicts.
```bash
conda create --name mecge python=3.9
conda activate mecge
```

2. **Install PyTorch:** Install PyTorch 2.2.2 from the official PyTorch website. Refer to the [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/) section for installation commands tailored to your system configuration (e.g., operating system, CUDA version).

3. **Install Required Packages:** Once the environment is set up and PyTorch is installed, install the necessary Python packages listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

4. **Install the Mamba Package:** Navigate to the mamba directory and install the package. This step ensures that all required components are properly installed.

```bash
cd mamba
pip install .
```

Note: Installing the package from the provided source (`mamba`) is recommended to prevent dependency issues and ensure compatibility across various packages. Follow these instructions carefully to avoid potential conflicts.

## Code Usage
- ### Training

1. **Set Hyperparameters:** Update the hyperparameters in the configuration file (.yaml) as needed.
2. **Run the Code:** Execute the following example command (e.g., the configuration file is `MECGE_phase.yaml`):
```bash
python main.py --n_type bw --config config/MECGE_phase.yaml
```

You can download the pre-trained model weights from this [link](https://drive.google.com/file/d/17qAyAJIw0zPFJwtkSsfwB7GeOsylq2_P/view?usp=sharing). Create a folder named model_weight and place the downloaded weight file inside it. We provide the best-performing models for various input features, with their results summarized in the table below.

| Model | Input | Loss function | SSD (au) $\downarrow$ | MAD (au) $\downarrow$ | PRD (%) $\downarrow$ | Cos_Sim $\uparrow$ |
|---    |---    |---            |---  |---  |---  |---      |
| <sub>MECG-E</sub> | <sub>Waveform</sub>   | $`\mathcal{L}_{time}`$</sub> | <sub>3.906 (6.662) | <sub>0.360 (0.281)</sub> | <sub>38.949 (22.947)</sub> | <sub>0.929 (0.081)</sub> |
| <sub>MECG-E</sub> | <sub>Complex</sub>    | $`\mathcal{L}_{time}`$+$`\mathcal{L}_{cpx}`$+$`\mathcal{L}_{con}`$ | <sub>3.891 (7.909)</sub> | <sub>0.326 (0.270)</sub> | <sub>37.734 (23.098)</sub> | <sub>0.931 (0.084)</sub> |
| <sub>MECG-E</sub> | <sub>Mag.+Phase</sub> | $`\mathcal{L}_{time}`$+$`\mathcal{L}_{cpx}`$+$`\mathcal{L}_{con}`$ | <sub>3.445 (6.493)</sub> | <sub>0.319 (0.252)</sub> | <sub>37.613 (22.389)</sub> | <sub>0.936 (0.077)</sub> |

- ### Testing
Training the model from scratch will automatically include the testing stage. For cases where the pretrained weights are already available (e.g., downloaded the pretrained weights), and retraining is not required, use the following command (e.g., the configuration file is `MECGE_phase.yaml`):

```bash
python main.py --n_type bw --config config/MECGE_phase.yaml --test
```

- ### Evaluate the Result
Evaluate the result of multiple configuration files (e.g., `MECGE_phase.yaml` and `MECGE_complex.yaml`).
```bash
python cal_metrics.py --experiments MECGE_phase MECGE_complex
```

Note: If you download the data from our provided [link](https://drive.google.com/file/d/19qOwywAoxreEv4xONTk-smQdo-ZdoPBc/view?usp=sharing), you can directly reference the results listed in Table below without additional training.

<p align="center">
<img src="figs/result.png"/>
</p>


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


## Citation

If you find the code helpful, please cite the following article.
```
@article{hung2024mecg,
  title={MECG-E: Mamba-based ECG Enhancer for Baseline Wander Removal},
  author={Hung, Kuo-Hsuan and Wang, Kuan-Chen and Liu, Kai-Chun and Chen, Wei-Lun and Lu, Xugang and Tsao, Yu and Lin, Chii-Wann},
  journal={arXiv preprint arXiv:2409.18828},
  year={2024}}

```

## Integration with PTB-XL Denoising Pipeline

MECG-E has been integrated into the PTB-XL ECG denoising benchmarking pipeline, allowing it to be trained and evaluated alongside other denoising models (FCN, UNet, IMUnet, Stage2/DRnet). This section documents the integration and how to use MECG-E within the parent repository.

### Overview

MECG-E can be used in two ways within the PTB-XL pipeline:
- **Standalone Stage1 denoiser:** Train and evaluate MECG-E independently
- **Stage1 component for Stage2:** Use MECG-E output as input to Stage2/DRnet refinement models

For complete documentation of the denoising pipeline, see: `../../0_README.md`

### Model Variants

Three MECG-E variants are available, each using different feature representations:

| Model Type | Features | Loss Function | Best For |
|-----------|----------|---------------|----------|
| `mecge_phase` | Magnitude + Phase STFT | $\mathcal{L}_{time}$ + $\mathcal{L}_{cpx}$ + $\mathcal{L}_{con}$ | Best overall performance |
| `mecge_complex` | Complex STFT | $\mathcal{L}_{time}$ + $\mathcal{L}_{cpx}$ + $\mathcal{L}_{con}$ | Good balance |
| `mecge_wav` | Waveform | $\mathcal{L}_{time}$ only | Fastest training, simplest |

Each variant is loaded as a separate model type in the pipeline. The **phase variant** (`mecge_phase`) typically achieves the best performance.

### Configuration

Configure MECG-E in the main denoising config file (`../../configs/denoising_config.yaml`):

```yaml
models:
  - name: "mecge_phase"
    type: "mecge_phase"  # Options: mecge_phase, mecge_complex, mecge_wav
    epochs: 50
    lr: 0.001
    batch_size: 32
```

The model `type` determines which YAML config file is loaded from the `config/` directory:
- `mecge_phase` → loads `config/MECGE_phase.yaml`
- `mecge_complex` → loads `config/MECGE_complex.yaml`
- `mecge_wav` → loads `config/MECGE_wav.yaml`

You can modify these YAML config files to adjust MECG-E hyperparameters (STFT parameters, Mamba blocks, loss weights, etc.).

### Input/Output Format

MECG-E has been adapted to accept the same input format as other Stage1 models:
- **Input shape:** `(batch, 1, 1, time)` where dimension 2 is a spatial dimension of size 1
- **Output shape:** `(batch, 1, 1, time)` matching the input format
- **Backward compatibility:** Also supports 3D input `(batch, 1, time)` for compatibility

The shape adaptation is handled automatically inside the model's `forward()` and `denoising()` methods. When 4D input is detected, the model:
1. Squeezes dimension 2 on input: `(batch, 1, 1, time)` → `(batch, 1, time)`
2. Processes with STFT + Mamba architecture
3. Unsqueezes dimension 2 on output: `(batch, 1, time)` → `(batch, 1, 1, time)`

### Training Interface

MECG-E uses a unique training interface compared to standard PyTorch models:

**Training mode:**
```python
loss = model.forward(clean_audio, noisy_audio)  # Returns scalar loss directly
```

**Inference mode:**
```python
denoised = model.denoising(noisy_audio)  # Returns denoised predictions
```

This differs from standard models where `forward()` returns predictions and loss is computed externally. The pipeline automatically detects MECG-E models using `hasattr(model, 'denoising')` and handles them appropriately.

**Users don't need to modify training code** - the detection is automatic.

### Loss Computation

MECG-E computes loss internally using multiple components:

**Phase and Complex variants:**
- **Time-domain loss:** MSE between clean and denoised waveforms
- **Complex loss:** MSE in STFT complex domain
- **Consistency loss:** Ensures STFT magnitude/phase consistency

**Waveform variant:**
- **Time-domain loss only:** Simple MSE for faster training

Loss weights are configured in the YAML config files (e.g., `time_weight`, `complex_weight`, `consistency_weight`).

### Usage Examples

#### Example 1: Train MECG-E standalone

Configure in `denoising_config.yaml`:
```yaml
models:
  - name: "mecge_phase"
    type: "mecge_phase"
    epochs: 50
    lr: 0.001
    batch_size: 32
```

Then run:
```bash
python train.py --config ../../configs/denoising_config.yaml
```

#### Example 2: Use MECG-E as Stage1 for Stage2 model

```yaml
models:
  # Stage 1: Train MECG-E
  - name: "mecge_phase"
    type: "mecge_phase"
    epochs: 50
    lr: 0.001
    batch_size: 32

  # Stage 2: Train DRnet using MECG-E output
  - name: "drnet_mecge"
    type: "stage2"
    stage1_model: "mecge_phase"  # Use MECG-E as Stage1
    epochs: 50
    lr: 0.001
    batch_size: 32
```

This combines MECG-E's strong denoising with Stage2 refinement for optimal results.

#### Example 3: Compare MECG-E variants

```yaml
models:
  - name: "mecge_phase"
    type: "mecge_phase"
    epochs: 50
    lr: 0.001
    batch_size: 32

  - name: "mecge_complex"
    type: "mecge_complex"
    epochs: 50
    lr: 0.001
    batch_size: 32

  - name: "mecge_wav"
    type: "mecge_wav"
    epochs: 50
    lr: 0.001
    batch_size: 32
```

### Evaluation

MECG-E is evaluated using the same metrics as other models:
- **SNR improvement (dB):** Output SNR - Input SNR
- **RMSE and RMSE improvement (%):** Percentage reduction in RMSE
- **Output SNR distribution:** Statistical analysis of denoising quality

Evaluation scripts automatically detect and handle MECG-E models.

**Similarity metrics:**
```bash
python evaluate_similarity.py --config ../../configs/denoising_config.yaml
```

**Downstream classification:**
```bash
python evaluate_downstream.py --config ../../configs/denoising_config.yaml
```

### Differences from Original MECG-E

Key differences between the integrated version and the original MECG-E:

| Aspect | Original MECG-E | Integrated Version |
|--------|-----------------|-------------------|
| **Input format** | 3D: `(batch, 1, time)` | 4D: `(batch, 1, 1, time)` (also supports 3D) |
| **Dataset** | QT Database + NSTDB noise | PTB-XL + online noise generation |
| **Noise generation** | Pre-corrupted dataset | Online with NoiseFactory |
| **Training pipeline** | Standalone script | Unified pipeline with other models |
| **Evaluation** | QT-specific metrics | PTB-XL metrics + downstream tasks |
| **Configuration** | Command-line args | YAML config integration |
| **Model loading** | Direct instantiation | Via `get_model()` utility |

**Note:** The core MECG-E architecture and loss functions remain unchanged.

### Technical Details

Implementation details for developers:

**Model loading:**
- `get_model()` in `../../denoising_utils/utils.py` loads MECG-E with appropriate YAML config
- Config path determined by model type: `mecge_phase` → `config/MECGE_phase.yaml`

**Training detection:**
- `train_model()` in `../../denoising_utils/training.py` uses `hasattr(model, 'denoising')` to detect MECG-E
- Calls `model(clean, noisy)` for MECG-E vs `model(noisy)` for standard models

**Inference:**
- `predict_with_model()` calls `model.denoising()` for MECG-E instead of `model()`
- Evaluation scripts (`qui_plot.py`, `rmse_analysis.py`, `evaluate_downstream.py`) all detect and handle MECG-E

**Key files:**
- Model implementation: `models/MECGE.py`
- Integration logic: `../../denoising_utils/utils.py`, `../../denoising_utils/training.py`
- Tests: `../../../../tests/test_mecge_forward.py`

### Pretrained Models

Pretrained MECG-E models can be loaded by specifying a path in the config:

```yaml
models:
  - name: "mecge_phase"
    type: "mecge_phase"
    pretrained_path: "path/to/pretrained/model.pth"  # Optional
    epochs: 50
    lr: 0.001
    batch_size: 32
```

**Important:** The original MECG-E pretrained weights (from `score/` directory) are trained on QT Database and may not transfer well to PTB-XL. We recommend **training MECG-E from scratch on PTB-XL** for best results.

### Troubleshooting

**Import errors:**
- Ensure the parent repository's path is in `sys.path`
- Check that Mamba package is installed: `cd mamba && pip install .`
- Verify Python >= 3.9 and CUDA >= 12.0

**Shape mismatch errors:**
- Verify input data has shape `(batch, 1, 1, time)` or `(batch, 1, time)`
- Check that the model's shape adaptation logic is working correctly
- Enable debug logging to see input/output shapes

**Loss computation errors:**
- Ensure YAML config files are present in `config/` directory
- Verify loss function parameters are correctly specified in config
- Check that STFT parameters (n_fft, hop_length) are compatible with signal length

**CUDA errors:**
- MECG-E requires CUDA >= 12.0 for Mamba operations
- Check CUDA version: `nvcc --version`
- Consider using CPU mode for testing (slower but no CUDA requirement):
  ```python
  device = torch.device('cpu')
  ```

**Performance issues:**
- Reduce batch size if encountering OOM errors
- Adjust STFT parameters (smaller n_fft) to reduce memory usage
- Consider using the `mecge_wav` variant for faster training

## Acknowledgments
* We acknowledge that our code is heavily based on implementations provided in several GitHub repositories ([DeepFilter](https://github.com/fperdigon/DeepFilter/tree/master/Data_Preparation), [DeScoD-ECG](https://github.com/HuayuLiArizona/Score-based-ECG-Denoising) and [SEMamba](https://github.com/RoyChao19477/SEMamba)), and we extend our gratitude to the respective authors for making their work publicly available.
* [Bio-ASP Lab](https://bio-asplab.citi.sinica.edu.tw), CITI, Academia Sinica, Taipei, Taiwan
