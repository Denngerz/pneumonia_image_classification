# configs/

Experiment settings. **This is the only place where experiments are allowed to differ.**

| File | Contents |
|---|---|
| `base.yaml` | Shared settings for every model: image size (224), batch size, epochs, optimizer, learning rate, schedule, early stopping, augmentation, seed |
| `models/<model>.yaml` | One tiny file per model containing **only** `model.name` (team decision: no per-model overrides; a test enforces it) |

Model configs: `baseline_cnn.yaml`, `vgg16.yaml`, `resnet50.yaml`, `densenet121.yaml`, `efficientnet_b0.yaml`,
`mobilenetv3_large.yaml`. The name must match an entry in `pneumonia/models.py`, which also pins the exact timm
ImageNet weight tags.

| Model | timm weights | Parameters |
|---|---|---|
| `baseline_cnn` | — (trained from scratch) | 0.39 M |
| `vgg16` | `vgg16.tv_in1k` | 134.3 M |
| `resnet50` | `resnet50.a1_in1k` | 23.5 M |
| `densenet121` | `densenet121.ra_in1k` | 7.0 M |
| `efficientnet_b0` | `efficientnet_b0.ra_in1k` | 4.0 M |
| `mobilenetv3_large` | `mobilenetv3_large_100.ra_in1k` | 4.2 M |

Training scripts load `base.yaml` first, then apply the model config on top. A copy of the final merged config is saved
with every run in `results/<run>/config.yaml`.
