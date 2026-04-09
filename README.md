# digitization-of-ECG-images

Training code for [PhysioNet Kaggle competition](https://www.kaggle.com/competitions/physionet-ecg-image-digitization).

## Files

| File           | Purpose                                      |
| -------------- | -------------------------------------------- |
| `config.py`    | Hyperparameters, paths, constants            |
| `dataset.py`   | Data loading, row cropping, augmentation     |
| `model.py`     | ResNet34 UNet + CoordConv + soft-argmax head |
| `loss.py`      | JSD + SNR combined loss                      |
| `metrics.py`   | Competition SNR metric                       |
| `train.py`     | Training loop with checkpoint resume         |
| `inference.py` | Prediction on test images                    |

## Usage

```python
# Train
from train import run_training
run_training()

# Resume
run_training(resume='checkpoints/best_fold0.pth')
```
