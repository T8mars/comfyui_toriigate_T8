# comfyui_toriigate_T8 Agent Runbook

本文件给后续协作 agent 使用。进入本仓库做代码、调试、UI、发布或文档任务时，先读本文件，再读 `README.md`、`nodes_api.py`、`nodes.py`、`js/` 下的前端脚本。

## 项目定位

- 这是 `T8mars/comfyui_toriigate_T8`，基于 `litch230/comfyui_toriigate` 改造。
- 目标是让 ToriiGate-0.5 在 ComfyUI 中优先通过本地 GGUF 运行。
- 默认推荐节点是 `ToriiGate Llama.cpp Vision Generate`，默认运行模式是 `local_gguf`。
- 外部 `llama-server` 模式仍保留为 `api_server`，但不是默认主路径。
- 原始 `ToriiGate Captioner` Transformers/PyTorch 节点保留，但不是当前 T8 主推路径。

## 仓库边界

- 当前仓库目录：`F:\AI-T8-video-onekey\ComfyUI\custom_nodes\comfyui_toriigate`
- 远端仓库：`https://github.com/T8mars/comfyui_toriigate_T8.git`
- 主分支：`main`
- 这是嵌在 ComfyUI 大目录里的独立 git 仓库；不要把外层 `F:\AI-T8-video-onekey\ComfyUI` 当成本项目仓库提交。
- 不要提交模型文件、缓存、`__pycache__`、`.tmp`、`.pytest_cache` 等产物。

## 关键文件

- `nodes_api.py`
  - Llama.cpp GGUF 本地模式和 API 模式都在这里。
  - `DEFAULT_LOCAL_MODEL_PATH` 和 `DEFAULT_LOCAL_MMPROJ_PATH` 指向默认 ToriiGate GGUF。
  - `_list_llm_gguf_paths()` 扫描 `ComfyUI/models/LLM`，把普通 GGUF 和 `mmproj` 分成两个下拉。
  - `_clean_float()` / `_clean_int()` 是后端 NaN/越界兜底，不要删。
  - `_build_local_chat_handler()` 优先使用 `Qwen3VLChatHandler`，并给 Qwen-VL 设置 `image_min_tokens=1024`。
- `nodes.py`
  - 原始 Grounding Builder 和 Transformers Captioner 在这里。
  - 文件底部把 `nodes_api.py` 的 API/GGUF 节点合并进 ComfyUI 的 node mapping。
- `js/toriigate_debug.js`
  - 目前不是调试空文件，而是数字 widget sanitizer。
  - 负责修复切换 tab 后 `Vision Generate` 等节点数字参数变成 `NaN` 的前端问题。
- `js/toriigate_grounding.js`
  - 只控制 Grounding Builder 的动态显示/隐藏，不应影响 Vision Generate。
- `requirements.txt`
  - 只保留通用依赖，不钉版本。
  - 不强制写入 `llama-cpp-python`，因为它需要按 CUDA / 平台选择 wheel。
- `README.md`
  - 面向用户的中文说明。模型下载、wheel 下载、使用方式应优先更新这里。

## 模型和依赖

默认模型路径：

```text
models/LLM/ToriiGate-0.5-Q8_0.gguf
models/LLM/ToriiGate-0.5-Q8_0.mmproj.gguf
```

模型下载地址：

```text
https://pan.quark.cn/s/cabaa25e75e5
```

llama-cpp-python wheel 下载地址：

```text
https://github.com/JamePeng/llama-cpp-python/releases
```

本机曾验证可用的 Python：

```text
F:\AI-T8-video-onekey\python\python.exe
```

本机曾验证 `llama_cpp 0.3.35` 可导入，且 `Qwen3VLChatHandler` 存在。

## 已修复的重要问题

- `Vision Generate` 默认不再请求 `http://127.0.0.1:8080/v1/chat/completions`，而是优先走 `local_gguf`。
- `models/LLM` 下普通 GGUF 和 `mmproj` 已自动下拉扫描，不需要用户手填路径。
- 文本 API 节点曾有 `actual_temperature` 未定义问题，已修。
- 切换 ComfyUI tab 后数字参数变 `NaN` 的问题已做双层兜底：
  - 前端：`js/toriigate_debug.js` 在节点创建、恢复、添加、绘制、queue 前清洗 widget。
  - 后端：`nodes_api.py` 在执行前用 `_clean_float()` / `_clean_int()` 清洗输入。
- 切换 tab 后出现 `decoding: <seed> not in ['sample', 'greedy_fast']` 的问题已修：
  - 根因是旧工作流/前端恢复按 widget 数组位置对齐，新增 `runtime/local_model_path/local_mmproj_path` 后可能产生参数位移。
  - `js/toriigate_debug.js` 现在会清洗 combo/string/bool/number，并识别旧版 Vision API widget 顺序做迁移。
  - `app.queuePrompt` 已加前置 sanitizer，避免节点在隐藏 tab 中未绘制时仍带错位参数入队。
- `seed` 上限已从 `0xFFFFFFFFFFFFFFFF` 降到 `0xFFFFFFFF`，避免 JS 数字精度/恢复问题。
- `requirements.txt` 已放松为无版本约束的通用依赖列表。

## 已知问题和待完善

- `ToriiGate Captioner` 的 `caption()` 开头引用了未定义的 `toriigate_model`：
  - 位置在 `nodes.py` 的 `ToriiGateCaptioner.caption()` 开头。
  - 这看起来是遗留/迁移错误，原始 Transformers 节点可能无法正常运行。
  - 若用户反馈 Captioner 报错，应优先修这个，而不是改 GGUF 节点。
- `local_gguf` 依赖 `llama-cpp-python` 的实际 wheel 能力：
  - 必须有 Qwen 视觉 handler，尤其是 `Qwen3VLChatHandler`。
  - 如果用户环境只有旧版 llama-cpp-python，视觉节点会报 handler 不存在。
- `api_server` 模式只适配 OpenAI 兼容的 `/v1/chat/completions`。
  - 用户的 8080 如果不是 llama-server 或没有该 endpoint，会 HTTP 404。
  - 默认建议切回 `local_gguf`。
- 修改前端 JS 后必须重启 ComfyUI；浏览器如缓存旧 JS，需要硬刷新。
- GitHub 页面可能显示本仓库比上游 fork ahead/behind。
  - 不要随手点 `Sync fork`。
  - 这是 T8 定制版，除非明确要吸收上游，否则保持当前主线。
- `_list_llm_gguf_paths()` 当前扫描 `models/LLM`。Windows 大小写不敏感；如果未来要兼容 Linux，可考虑同时处理 `models/llm`。

## 验证命令

在仓库根目录运行：

```powershell
& "F:\AI-T8-video-onekey\python\python.exe" -m py_compile .\__init__.py .\init.py .\nodes.py .\nodes_api.py .\prompts.py
```

JS 语法检查：

```powershell
node --check .\js\toriigate_debug.js
node --check .\js\toriigate_grounding.js
```

检查本机 llama-cpp-python：

```powershell
& "F:\AI-T8-video-onekey\python\python.exe" -c "import llama_cpp, llama_cpp.llama_chat_format as f; print(llama_cpp.__version__); print(hasattr(f, 'Qwen3VLChatHandler'))"
```

检查本地模型下拉扫描：

```powershell
& "F:\AI-T8-video-onekey\python\python.exe" -c "import sys; sys.path.insert(0, r'F:\AI-T8-video-onekey\ComfyUI'); sys.path.insert(0, r'F:\AI-T8-video-onekey\ComfyUI\custom_nodes'); import comfyui_toriigate.nodes_api as n; print(n._list_llm_gguf_paths(mmproj=False)[:3]); print(n._list_llm_gguf_paths(mmproj=True)[:3])"
```

## Git 和发布

由于 Windows ownership / sandbox 限制，git 命令常需要带：

```powershell
git -c safe.directory=F:/AI-T8-video-onekey/ComfyUI/custom_nodes/comfyui_toriigate ...
```

发布前检查：

```powershell
git -c safe.directory=F:/AI-T8-video-onekey/ComfyUI/custom_nodes/comfyui_toriigate status -sb
git -c safe.directory=F:/AI-T8-video-onekey/ComfyUI/custom_nodes/comfyui_toriigate ls-remote origin refs/heads/main
```

常规推送到 main：

```powershell
git -c safe.directory=F:/AI-T8-video-onekey/ComfyUI/custom_nodes/comfyui_toriigate add <files>
git -c safe.directory=F:/AI-T8-video-onekey/ComfyUI/custom_nodes/comfyui_toriigate commit -m "<message>"
git -c safe.directory=F:/AI-T8-video-onekey/ComfyUI/custom_nodes/comfyui_toriigate push origin main
```

如果不是用户明确要求覆盖，不要 force push。

## 协作注意事项

- 开始任务前先读 `SKILL.md`、`README.md`、`nodes_api.py`、相关 JS。
- 不要把 README 当作唯一事实来源；代码优先。
- 用户偏好直接可用的本地 ComfyUI 节点，优先修工作流实际报错。
- 对本地 Windows 整合包任务，优先使用 `F:\AI-T8-video-onekey\python\python.exe` 验证。
- 代码改动后至少跑 `py_compile`；涉及 JS 时再跑 `node --check`。
- 涉及前端 UI 的问题，提醒用户重启 ComfyUI 和硬刷新浏览器。
- 依赖不要随便钉版本；`llama-cpp-python` 不要写死在 `requirements.txt`，除非用户明确要求。
- 继续维护 README 时保持中文主文档，并保留上游项目/模型来源链接。
