# 训练模板
自用训练模板，包括通用模型、LLM Lora和CV

## 模板选择
| 模板 | 适用 | 训练内核 | 冒烟 |
|---|---|---|---|
| general | 表格/自定义MLP | 自写 fit | sklearn toy |
| cv | 图像分类 | 复用fit | FakeData + resnet18 |
| llm_lora | LLM LoRA 微调 | Trainer | tiny-gpt2 |

## 公共层 common/
- config.py：加载 + 类型转换
- utils.py：seed / device / dir
- checkpoint.py：state_dict + adapter
- trainer.py：fit
- plot.py：history / loss_curve / confusion_matrix

## 环境
每个模板独立 venv + kernel，requirements.txt