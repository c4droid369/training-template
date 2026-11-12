# LLM LoRA 微调模板

## 适用场景
- 因果语言模型 LoRA 微调
- 外层函数式编排，训练内核用 `transformers.Trainer`
- 含 adapter 保存、加载、生成推理、loss 曲线

## 目录
```text
llm_lora/
├── requirements.txt
├── config.yaml
├── main.ipynb
└── outputs/            # adapter/、loss.png
```

## 环境
```bash
python -m venv .venv-llm
source .venv-llm_lora/bin/activate
pip install -r llm_lora/requirements.txt
python -m ipykernel install --user --name llm --display-name "Python (llm)"
```
kernel 选 **Python (llm)**

## 快速运行（冒烟）
1. 默认 `model_name: sshleifer/tiny-gpt2`、`max_length: 64`、`batch_size: 2`、`epochs: 1`
2. Restart & Run All
3. 验收：`outputs/adapter/` 有 `adapter_config.json` 和 `adapter_model.safetensors`；`generate()` 能输出文本；`outputs/loss.png` 生成

## 关键入口
| 想改什么 | 改哪里 |
|---|---|
| 换基座模型 | `config.yaml` 的 `model_name` |
| LoRA 目标层 | `config.yaml` 的 `target_modules`（tiny-gpt2 用 `c_attn`，LLaMA 系用 `q_proj/v_proj`） |
| 数据 | `make_toy_dataset` 换成真实 instruction 数据 |
| 训练参数 | `TrainingArguments` 所在函数 |
| 生成参数 | `generate()` 的 `max_new_tokens / do_sample` |

## 配置注意
- tokenizer 没有 `pad_token` 时必须设 `pad_token = eos_token`，否则 `generate` 报错
- `TrainingArguments` 用 `use_cpu=True`、`report_to=[]`、`save_strategy="no"`，冒烟别开 fp16
- `plot_loss_curve` 的数据来自 `trainer.state.log_history` 中带 `loss` 的条目
- `out_dir` 由 notebook 定义后传给 `run_training`，别依赖全局

## 输出物
- `outputs/adapter/`：LoRA adapter
- `outputs/loss.png`：训练 loss 曲线

## 已知限制
- 真实微调需 GPU，并调大 `batch_size`、`max_length`、`epochs`
- QLoRA 需 `bitsandbytes`，CPU 不可用，模板默认不含
- 不同模型 `target_modules` 不同，换模型必须核对