# CLAUDE.md

本仓库是 FunAudioLLM/CosyVoice 的个人 fork（`origin`=GitHub，`gitea`=内网 192.168.8.99，`upstream`=官方），
在官方代码上加了改进版 WebUI、REST API、音色库、多音字处理和 RTX 5090 适配。回复与注释用简体中文。

## 运行环境
- conda 环境 `cosyvoice`：`/home/user1/anaconda3/envs/cosyvoice/bin/python`（Python 3.12，CUDA 12.8，RTX 5090）。
  系统默认 `python` 不是这个环境，跑脚本要用全路径。
- 依赖用 `req_5090.txt`（新栈 torch 2.11 / cu128）；官方 `requirements.txt` 是旧栈，不要用于本机。
- 文本前端走 `wetext`（没装 `ttsfrd`）。
- onnxruntime 只能装 `onnxruntime-gpu`；CPU 版（faster-whisper 会拉进来）会覆盖它导致只剩 CPU 后端，修法见 `req_5090.txt`。
- 主力模型 `pretrained_models/Fun-CosyVoice3-0.5B`；`pretrained_models/`、`outputs/`、`spk_library/` 不入库。

## 启动
```bash
# 统一服务：REST API(/api) + Gradio WebUI(/ui)，同一进程共用模型
python cosyvoice_api.py --model_dir pretrained_models/Fun-CosyVoice3-0.5B --port 50000
# 仅 WebUI
python webui.py --model_dir pretrained_models/Fun-CosyVoice3-0.5B --port 8007
```
服务通常由用户在自己终端前台运行，不要擅自杀掉或重启；改了代码要提醒用户重启。
测试时另起脚本加载模型（显存 32G，模型约 4G，够用）。

## 自定义代码（官方代码以外）
- `webui.py`：核心逻辑都在这里——模型初始化 `init_runtime()`、推理入口 `build_iterator()`、
  音色库（`spk_library/<模型目录名>/<name>.pt|.wav|.json`）、输出历史（`outputs/`）、推理锁 `INFER_LOCK`。
  CosyVoice3 需要的 `You are a helpful assistant.<|endofprompt|>` 前缀由 `fix_*_text()` 自动补全。
- `cosyvoice_api.py`：FastAPI 封装，直接 `import webui as W` 复用其函数，不要重复实现逻辑。
  `ai-video-studio/tts_service/cosyvoice_api.py` 是同一文件的副本，改动后需同步。
- `polyphone.py` + `polyphone_dict.txt`：多音字。见下节。

## 多音字 / 指定读音
CosyVoice3 词表含拼音 token（声母 `[y]` + 带调韵母 `[uè]`），用它替换汉字即可强制读音：`不亦[y][uè]乎`。
- 用户写法：`字{拼音}`，如 `不亦说{yuè}乎` / `说{yue4}` / `绿{lv4}`，由 `polyphone.apply_pinyin()` 转换并按词表校验，
  不支持的拼音抛 `ValueError`（WebUI 弹错、API 返回 400）。
- `polyphone_dict.txt`：每行一个带标注的词句，出现原文即自动套用，长词优先，按 mtime 热加载。
  只收确定无争议的读音；有歧义、依上下文或新旧读音有争议的不要加。
- 在 `build_iterator()` 开头统一处理；非 v3 模型只去掉标注。手动标注优先，词典不跨过已标注的字匹配。
- 多音字检查：`polyphone.analyze()`（WebUI「多音字检查」、`POST /api/pinyin`）列出常见多音字及来源。
  pypinyin 推测只做提示、**不要自动强制读音**——它会出错（一骑→qí、行人→háng），而模型按上下文通常读对。
- 验证读音/字错率可用 faster-whisper（`large-v3`；传 numpy 数组，av 版本不兼容文件输入；
  GPU 需 `LD_LIBRARY_PATH=<env>/lib/python3.12/site-packages/nvidia/{cublas,cudnn}/lib`）：
  yuè 会被识别成"跃/悦"，shuō 识别成"说"。

## 性能结论（RTX 5090，实测）
- RTF ≈ 0.2；LLM 自回归占约 90% 耗时，每 token ~7ms 且受 CPU/kernel launch 限制（GPU 实际只占 ~2ms）。
- flow 的 TensorRT 无收益（已让转换兼容 TRT 11，但默认不开）；`--fp16` 反而更慢（autocast 开销）。
- 已做：LLM 单步跳过 lm_head（输出逐位一致，+3~7%）。HF StaticCache + CUDA Graph 试过，
  fp32 下 SDPA 选到很慢的 mem-efficient kernel，且 transformers 5.x 下有越界断言，放弃。
  真正提速需要 vLLM（官方支持，但需独立环境，避免破坏 torch 版本）。
- `--llm rl` 可切 `llm.rl.pt`；自测 CER/SIM 与 base 差异在噪声内，默认 base。
- 长文本分段（约 80 token 一段）接缝自然（段首尾各有静音、无爆音），无需额外处理。

## 注意
- 文本含 `<|` `|>` 时 frontend 会跳过文本正则化（见 `cosyvoice/cli/frontend.py:text_normalize`）。
- 修改官方目录 `cosyvoice/` 下代码要克制，便于以后合并 upstream。
