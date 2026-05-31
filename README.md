# comfyui_toriigate_T8

这是 ToriiGate-0.5 的 ComfyUI 自定义节点 T8 版本。这个版本保留原有
Transformers 节点，同时新增了 `llama-cpp-python` 本地 GGUF 直连模式：
不需要手动启动 `llama-server`，节点可以直接扫描并加载
`ComfyUI/models/LLM` 下的本地模型。

基于以下项目改造：

- 原始节点项目：[litch230/comfyui_toriigate](https://github.com/litch230/comfyui_toriigate)
- 原始模型：[Minthy/ToriiGate-0.5](https://huggingface.co/Minthy/ToriiGate-0.5)
- GGUF 模型：[DraconicDragon/ToriiGate-0.5-GGUF](https://huggingface.co/DraconicDragon/ToriiGate-0.5-GGUF)

## 主要改动

- `ToriiGate Llama.cpp Vision Generate` 默认使用 `local_gguf` 本地运行模式。
- 节点会自动扫描 `ComfyUI/models/LLM` 目录下的 `.gguf` 模型。
- 普通 GGUF 模型和 `mmproj` 视觉投影模型会分成两个下拉框选择。
- 默认优先选择 `ToriiGate-0.5-Q8_0.gguf` 和 `ToriiGate-0.5-Q8_0.mmproj.gguf`。
- 仍保留外部 `llama-server` 的 OpenAI 兼容 API 模式，可切换为 `api_server`。
- 修复文本 API 节点中 `actual_temperature` 未定义的问题。

## 安装

把仓库克隆到 `ComfyUI/custom_nodes`：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/comfyui_toriigate_T8.git
```

安装或更新后重启 ComfyUI。

## 模型放置位置

把 GGUF 模型放到：

```text
ComfyUI/models/LLM/
```

推荐默认文件名：

```text
ComfyUI/models/LLM/ToriiGate-0.5-Q8_0.gguf
ComfyUI/models/LLM/ToriiGate-0.5-Q8_0.mmproj.gguf
```

节点会递归扫描 `ComfyUI/models/LLM`：

- `local_model_path`：列出文件名不包含 `mmproj` 的 `.gguf` 文件。
- `local_mmproj_path`：列出文件名包含 `mmproj` 的 `.gguf` 文件。

如果上面两个 ToriiGate 文件存在，它们会自动排在下拉框第一项。

## 节点说明

### ToriiGate Grounding Builder

用于生成 ToriiGate 所需的提示词文本。可选择官方常用格式，例如
`long_thoughts_v2`、`json`、`short`、漫画相关格式等。

### ToriiGate Llama.cpp Vision Generate

推荐使用的 GGUF 图片反推节点。

常用参数：

- `runtime`：默认 `local_gguf`，直接在 ComfyUI 内加载本地 GGUF；切到 `api_server` 时使用外部 `llama-server`。
- `local_model_path`：从 `ComfyUI/models/LLM` 自动扫描出的主 GGUF 模型。
- `local_mmproj_path`：从 `ComfyUI/models/LLM` 自动扫描出的视觉投影 GGUF。
- `chat_handler`：默认 `auto` 即可，会优先使用可用的 Qwen 视觉 handler。
- `keep_model_alive`：连续生成时建议开启；关闭后每次生成结束释放本地模型。
- `n_gpu_layers`：默认 `-1`，让 llama.cpp 尽可能把层卸载到 GPU。
- `n_ctx`：上下文长度，默认 `8192`。

### ToriiGate Llama.cpp Text Generate

文本生成辅助节点。也支持 `local_gguf`，并从 `ComfyUI/models/LLM` 自动列出
本地 GGUF 模型。

### ToriiGate Captioner

原始 Transformers/PyTorch 本地节点，加载 Hugging Face 原始模型格式，不支持
GGUF。

## 本地 GGUF 模式

默认模式为：

```text
runtime = local_gguf
```

默认加载：

```text
models/LLM/ToriiGate-0.5-Q8_0.gguf
models/LLM/ToriiGate-0.5-Q8_0.mmproj.gguf
```

这些是相对 ComfyUI 根目录的路径，实际指向：

```text
<ComfyUI>/models/LLM/ToriiGate-0.5-Q8_0.gguf
<ComfyUI>/models/LLM/ToriiGate-0.5-Q8_0.mmproj.gguf
```

本地模式要求启动 ComfyUI 的 Python 环境中安装了较新的
`llama-cpp-python`，并且支持 Qwen 视觉/mmproj。

可用下面命令检查：

```bash
python -c "import llama_cpp, llama_cpp.llama_chat_format as f; print(llama_cpp.__version__); print(hasattr(f, 'Qwen3VLChatHandler'))"
```

输出里第二行是 `True` 时，本地视觉 GGUF 模式可用。

## 外部 API 模式

如果你想继续使用单独启动的 `llama-server`，把节点的 `runtime` 改为：

```text
api_server
```

示例启动命令：

```cmd
llama-server.exe ^
  --model ToriiGate-0.5-Q8_0.gguf ^
  --mmproj ToriiGate-0.5-Q8_0.mmproj.gguf ^
  --port 8080 ^
  -b 2048 -ub 1024 -fa on -fit on -fitt 1024 -ngl 999
```

节点里的 `server_url` 填：

```text
http://127.0.0.1:8080
```

API 模式会请求：

```text
http://127.0.0.1:8080/v1/chat/completions
```

如果你的服务没有这个 OpenAI 兼容接口，就会出现 HTTP 404。此时建议直接使用
默认的 `local_gguf` 模式。

## Python 依赖

原始 Transformers 节点需要：

```bash
pip install -r requirements.txt
```

本地 GGUF 模式需要 `llama-cpp-python`。不同显卡和后端的安装方式不同，
很多 ComfyUI 整合包已经内置这个依赖。如果你的 ComfyUI Python 不能
`import llama_cpp`，请把 `llama-cpp-python` 安装到启动 ComfyUI 的同一个
Python 环境中。

## 常见问题

### 节点还是显示旧参数或没有模型下拉框

重启 ComfyUI。节点输入项是在 ComfyUI 加载 custom node 时生成的。

### `/v1/chat/completions` 返回 HTTP 404

说明当前使用的是 `api_server`，但外部服务没有 OpenAI 兼容 chat endpoint。
切回 `local_gguf`，或启动支持该接口的新版 `llama-server`。

### 找不到本地模型

确认文件放在：

```text
ComfyUI/models/LLM/
```

如果刚复制模型文件，需要重启 ComfyUI 才能刷新下拉框。

### `llama_cpp` 未安装

把 `llama-cpp-python` 安装到启动 ComfyUI 的 Python 环境，然后重启 ComfyUI。

### 选错了 mmproj

主模型和 mmproj 要匹配。默认 ToriiGate Q8 模型应搭配：

```text
models/LLM/ToriiGate-0.5-Q8_0.mmproj.gguf
```

## 适用场景

ToriiGate-0.5 适合动漫、插画、数字艺术图片的反推描述、数据集 caption、
标签整理和 ComfyUI 工作流中的提示词辅助生成。
