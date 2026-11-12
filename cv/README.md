# CV 图像分类模板

## 适用场景
- 图像分类（ResNet / CNN / 可换 ViT）
- 复用 `common/trainer.py` 的 `fit`，只重写数据与模型
- 含混淆矩阵、样本网格、单图推理

## 目录
```text
cv/
├── requirements.txt
├── config.yaml
├── main.ipynb
└── outputs/            # best.pt、curves.png、confusion.png
```

## 环境
```bash
python -m venv .venv-cv
source .venv-cv/bin/activate
pip install -r cv/requirements.txt
python -m ipykernel install --user --name cv --display-name "Python (cv)"
```
kernel 选 **Python (cv)**。`torchvision` 由 pip 自动匹配 torch 版本。

## 快速运行（冒烟）
1. `config.yaml` 默认 `n_samples: 64`、`image_size: 64`、`batch_size: 8`、`pretrained: false`
2. Restart & Run All
3. 验收：`best.pt / curves.png / confusion.png` 生成；单图推理返回 pred + confidence

## 关键入口
| 想改什么 | 改哪里 |
|---|---|
| 换真实数据 | `make_loaders` 改成 `ImageFolder(root, transform=...)` |
| 换模型 | `build_model`，如 `timm.create_model` 或其他 torchvision 模型 |
| 预处理 | `make_loaders` 与 `predict_one` **必须保持一致** |
| 预训练权重 | `config.yaml` 的 `pretrained` |

## 配置注意
- 冒烟保持 `pretrained: false`，避免下载
- `num_classes` 要和数据类别数一致

## 输出物
- `outputs/best.pt`
- `outputs/curves.png`
- `outputs/confusion.png`

## 已知限制
- 冒烟用 `torchvision.datasets.FakeData`，不是真实图片
- CPU 上 resnet18 只适合小图小样本
- 第一版不加 `Normalize`，换真实数据时建议在 `make_loaders` 和 `predict_one` 同步加
```