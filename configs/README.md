# configs/

Experiment settings. **This is the only place where experiments are allowed to differ.**

| File | Contents |
|---|---|
| `base.yaml` | Shared settings for every model: image size (224), batch size, epochs, optimizer, learning rate, schedule, early stopping, augmentation, seed |
| `models/<model>.yaml` | One small file per model: the model name and only the overrides that are truly model-specific (should be rare) |

Planned model configs: `baseline_cnn.yaml`, `vgg16.yaml`, `resnet50.yaml`, `densenet121.yaml`,
`efficientnet_b0.yaml`, `mobilenetv3_large.yaml` (optional: `convnext_tiny.yaml`).

Training scripts load `base.yaml` first, then apply the model config on top. A copy of the final merged config is saved
with every run in `results/<run>/config.yaml`.
