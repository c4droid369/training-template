# 通用 PyTorch 训练模板

## 适用场景
- 表格 / 向量特征分类（MLP）
- 自定义小模型、快速验证训练流程
- 作为其他模板的底座：config / utils / checkpoint / trainer / plot 均来自 `common/`

## 目录
```text
general/
├── requirements.txt
├── config.yaml
├── main.ipynb
└── outputs/            # 运行后生成 best.pt、curves.png
```

## 环境
```bash
python -m venv .venv-general
source .venv-general/bin/activate
pip install -r general/requirements.txt
python -m ipykernel install --user --name general --display-name "Python (general)"
```
notebook kernel 选 **Python (general)**。

## 快速运行（冒烟）
1. 打开 `main.ipynb`
2. 顶部参数区可覆盖：`cfg = with_overrides(cfg, epochs=3)`
3. Restart & Run All
4. 验收：`outputs/best.pt`、`outputs/curves.png` 生成

## 关键入口
| 想改什么 | 改哪里 |
|---|---|
| 数据 | notebook 的 `make_toy_arrays` / `make_loaders` |
| 模型 | `build_model(cfg)` |
| 超参 | `config.yaml` |
| 指标/训练逻辑 | `common/trainer.py` 的 `train_one_epoch/evaluate/fit` |

## 输出物
- `outputs/best.pt`：验证集 best 权重
- `outputs/curves.png`：loss/acc 曲线

## 已知限制
- toy 数据来自`sklearn.make_classification`