"""
ToriiGate – llama.cpp nodes
===========================
These nodes can run a local GGUF model directly through llama-cpp-python, or
communicate with a llama-server (llama.cpp) that exposes an OpenAI-compatible
HTTP API. They do NOT import transformers, torch model loaders, AutoProcessor,
AutoModel, or any CUDA initialisation code.

Typical server launch (example):
    llama-server.exe ^
        --model ToriiGate-0.5-Q4_K_M.gguf ^
        --mmproj mmproj-ToriiGate-0.5-f16.gguf ^
        --port 8000

When runtime is api_server, the nodes send HTTP requests to
http://127.0.0.1:8000/v1/chat/completions.
"""

import base64
import gc
import io
import json
import logging
import math
import os
import random
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image

from .nodes import CAPTION_TYPES, _empty_grounding
from .prompts import make_user_query, prompts_b, prompts_names_only, system_prompt

logger = logging.getLogger("ToriiGate.API")


# ---------------------------------------------------------------------------
# Shared helper functions
# ---------------------------------------------------------------------------

DEFAULT_LOCAL_MODEL_PATH = "models/LLM/ToriiGate-0.5-Q8_0.gguf"
DEFAULT_LOCAL_MMPROJ_PATH = "models/LLM/ToriiGate-0.5-Q8_0.mmproj.gguf"
DEFAULT_IMAGE_MIN_TOKENS = 1024

_LOCAL_LLAMA_CACHE = {}
_LOCAL_LLAMA_LOCK = Lock()


def _clean_float(value, default: float, min_value: float | None = None, max_value: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)

    if not math.isfinite(number):
        number = float(default)

    if min_value is not None:
        number = max(float(min_value), number)
    if max_value is not None:
        number = min(float(max_value), number)
    return number


def _clean_int(value, default: int, min_value: int | None = None, max_value: int | None = None) -> int:
    number = int(round(_clean_float(value, float(default), min_value, max_value)))
    if min_value is not None:
        number = max(int(min_value), number)
    if max_value is not None:
        number = min(int(max_value), number)
    return number


def _comfyui_root() -> Path:
    try:
        import folder_paths

        base_path = getattr(folder_paths, "base_path", None)
        if base_path:
            return Path(base_path).resolve()

        models_dir = getattr(folder_paths, "models_dir", None)
        if models_dir:
            return Path(models_dir).resolve().parent
    except Exception:
        pass

    return Path(__file__).resolve().parents[2]


def _models_dir() -> Path:
    try:
        import folder_paths

        models_dir = getattr(folder_paths, "models_dir", None)
        if models_dir:
            return Path(models_dir).resolve()
    except Exception:
        pass

    return _comfyui_root() / "models"


def _list_llm_gguf_paths(*, mmproj: bool) -> list[str]:
    default_path = DEFAULT_LOCAL_MMPROJ_PATH if mmproj else DEFAULT_LOCAL_MODEL_PATH
    llm_dir = _models_dir() / "LLM"
    if not llm_dir.exists():
        return [default_path]

    paths = []
    for file_path in sorted(llm_dir.rglob("*.gguf"), key=lambda path: str(path).lower()):
        is_mmproj = "mmproj" in file_path.name.lower()
        if is_mmproj != mmproj:
            continue
        rel_path = file_path.relative_to(llm_dir).as_posix()
        paths.append(f"models/LLM/{rel_path}")

    if default_path in paths:
        paths.remove(default_path)
        paths.insert(0, default_path)
    elif not paths:
        paths.append(default_path)

    return paths


def _resolve_comfy_relative_path(path_text: str, default_path: str, label: str) -> Path:
    raw_path = (path_text or default_path).strip()
    if not raw_path:
        raise RuntimeError(f"[ToriiGate Local] {label} path is empty.")

    expanded = Path(os.path.expandvars(raw_path)).expanduser()
    if expanded.is_absolute():
        candidates = [expanded]
    else:
        parts = [part.lower() for part in expanded.parts]
        root = _comfyui_root()
        model_root = _models_dir()
        candidates = []

        if parts and parts[0] == "models":
            candidates.append(root / expanded)
        else:
            if len(expanded.parts) == 1:
                candidates.append(model_root / "LLM" / expanded)
            candidates.append(model_root / expanded)
            candidates.append(root / expanded)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise RuntimeError(
        f"[ToriiGate Local] Cannot find {label} file '{raw_path}'.\n"
        f"Checked:\n{searched}"
    )


def _extract_chat_text(response: dict) -> str:
    try:
        choice = response["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"[ToriiGate Local] Unexpected llama-cpp-python response:\n"
            f"{json.dumps(response, ensure_ascii=False, indent=2)[:500]}"
        ) from exc

    if isinstance(choice, dict):
        message = choice.get("message") or {}
        text = message.get("content", choice.get("text", ""))
    else:
        message = getattr(choice, "message", None)
        text = getattr(message, "content", "") if message is not None else getattr(choice, "text", "")

    if isinstance(text, list):
        text = "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in text)

    return str(text)


def _build_local_chat_handler(mmproj_path: Path, chat_handler: str, verbose: bool):
    try:
        import llama_cpp.llama_chat_format as chat_formats
    except ImportError as exc:
        raise RuntimeError(
            "[ToriiGate Local] llama-cpp-python is installed, but its "
            "llama_chat_format module is unavailable. Upgrade llama-cpp-python "
            "or use runtime='api_server'."
        ) from exc

    candidates_by_mode = {
        "auto": [
            "Qwen35VLChatHandler",
            "Qwen3VLChatHandler",
            "Qwen25VLChatHandler",
            "Qwen2VLChatHandler",
            "Llava15ChatHandler",
        ],
        "qwen35-vl": ["Qwen35VLChatHandler", "Qwen3VLChatHandler"],
        "qwen2.5-vl": ["Qwen25VLChatHandler", "Qwen2VLChatHandler"],
        "llava-1.5": ["Llava15ChatHandler"],
    }
    handler_names = candidates_by_mode.get(chat_handler, candidates_by_mode["auto"])
    last_error = None

    for handler_name in handler_names:
        handler_cls = getattr(chat_formats, handler_name, None)
        if handler_cls is None:
            continue

        for kwargs in (
            {
                "clip_model_path": str(mmproj_path),
                "verbose": verbose,
                "image_min_tokens": DEFAULT_IMAGE_MIN_TOKENS,
            },
            {"clip_model_path": str(mmproj_path)},
        ):
            try:
                print(f"[ToriiGate Local] Using {handler_name} with mmproj: {mmproj_path}")
                return handler_cls(**kwargs)
            except TypeError as exc:
                last_error = exc

    available = ", ".join(
        name for name in candidates_by_mode["auto"] if hasattr(chat_formats, name)
    ) or "none"
    detail = f"\nLast error: {last_error}" if last_error else ""
    raise RuntimeError(
        "[ToriiGate Local] No compatible vision chat handler was found in "
        "llama-cpp-python. ToriiGate GGUF is a Qwen3.5 vision model, so this "
        "usually means llama-cpp-python is too old. Upgrade it, or switch the "
        "node runtime to 'api_server' and run a recent llama-server with "
        f"--mmproj.\nAvailable vision handlers: {available}{detail}"
    )


def _get_local_llama(
    model_path: Path,
    mmproj_path: Path | None,
    chat_handler: str,
    n_ctx: int,
    n_gpu_layers: int,
    n_threads: int,
    verbose: bool,
):
    cache_key = (
        str(model_path),
        str(mmproj_path) if mmproj_path else "",
        chat_handler if mmproj_path else "",
        int(n_ctx),
        int(n_gpu_layers),
        int(n_threads),
    )

    with _LOCAL_LLAMA_LOCK:
        if cache_key in _LOCAL_LLAMA_CACHE:
            print(f"[ToriiGate Local] Reusing cached GGUF model: {model_path.name}")
            return _LOCAL_LLAMA_CACHE[cache_key], cache_key

        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "[ToriiGate Local] The Python package 'llama_cpp' is not installed "
                "in the Python environment that is running ComfyUI. Install a recent "
                "llama-cpp-python build with Qwen3.5/mmproj support, or switch the "
                "node runtime to 'api_server'."
            ) from exc

        local_chat_handler = None
        if mmproj_path is not None:
            local_chat_handler = _build_local_chat_handler(mmproj_path, chat_handler, verbose)

        kwargs = {
            "model_path": str(model_path),
            "n_ctx": int(n_ctx),
            "n_gpu_layers": int(n_gpu_layers),
            "verbose": bool(verbose),
        }
        if int(n_threads) > 0:
            kwargs["n_threads"] = int(n_threads)
        if local_chat_handler is not None:
            kwargs["chat_handler"] = local_chat_handler

        print(
            f"[ToriiGate Local] Loading GGUF model: {model_path} "
            f"(mmproj={mmproj_path or 'none'}, n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers})."
        )
        llm = Llama(**kwargs)
        _LOCAL_LLAMA_CACHE[cache_key] = llm
        return llm, cache_key


def _drop_local_llama(cache_key) -> None:
    with _LOCAL_LLAMA_LOCK:
        llm = _LOCAL_LLAMA_CACHE.pop(cache_key, None)
    if llm is not None:
        del llm
        gc.collect()


def run_local_chat_completion(
    *,
    messages,
    model_path: str,
    mmproj_path: str | None,
    chat_handler: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    n_ctx: int,
    n_gpu_layers: int,
    n_threads: int,
    keep_model_alive: bool,
    verbose: bool,
) -> str:
    resolved_model_path = _resolve_comfy_relative_path(
        model_path, DEFAULT_LOCAL_MODEL_PATH, "GGUF model"
    )
    resolved_mmproj_path = None
    if mmproj_path is not None:
        resolved_mmproj_path = _resolve_comfy_relative_path(
            mmproj_path, DEFAULT_LOCAL_MMPROJ_PATH, "GGUF mmproj"
        )

    llm, cache_key = _get_local_llama(
        model_path=resolved_model_path,
        mmproj_path=resolved_mmproj_path,
        chat_handler=chat_handler,
        n_ctx=int(n_ctx),
        n_gpu_layers=int(n_gpu_layers),
        n_threads=int(n_threads),
        verbose=bool(verbose),
    )

    kwargs = {
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    if int(seed) != 0:
        kwargs["seed"] = int(seed)

    try:
        try:
            response = llm.create_chat_completion(**kwargs)
        except TypeError as exc:
            if "seed" not in kwargs or "seed" not in str(exc):
                raise
            kwargs.pop("seed", None)
            response = llm.create_chat_completion(**kwargs)

        text = _extract_chat_text(response)
        logger.info("[ToriiGate Local] Received %d characters of generated text.", len(text))
        return text
    finally:
        if not keep_model_alive:
            _drop_local_llama(cache_key)


def image_tensor_to_base64(image_tensor, max_pixels_mp: float = 1.0) -> str:
    """Convert a ComfyUI IMAGE tensor (B, H, W, C float32 in [0,1]) to a
    base64-encoded PNG string suitable for embedding in a data-URI.
    Downscales the image if it exceeds max_pixels_mp to prevent massive TTFT."""
    img_np = (image_tensor[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")

    current_pixels = img_pil.width * img_pil.height
    max_pixels_count = max_pixels_mp * 1_000_000
    if current_pixels > max_pixels_count:
        scale = (max_pixels_count / current_pixels) ** 0.5
        new_w = max(1, int(img_pil.width * scale))
        new_h = max(1, int(img_pil.height * scale))
        logger.info(f"[ToriiGate API] Downscaling image from {img_pil.width}x{img_pil.height} to {new_w}x{new_h}")
        img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_vision_payload(
    model_name: str,
    system_prompt: str,
    user_text: str,
    image_b64: str,
    temperature: float,
    max_tokens: int,
    seed: int = 0,
) -> dict:
    """Build an OpenAI-compatible chat-completions payload with an image."""
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})

    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
                {"type": "text", "text": user_text},
            ],
        }
    )

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed:
        payload["seed"] = seed
    return payload


def build_text_payload(
    model_name: str,
    system_prompt: str,
    user_text: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    """Build an OpenAI-compatible chat-completions payload without an image."""
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})

    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    return payload


def send_chat_request(server_url: str, payload: dict, timeout: float) -> str:
    """POST *payload* to ``{server_url}/v1/chat/completions`` and return the
    generated text.  Raises a ``RuntimeError`` with a human-readable message
    on any failure so that ComfyUI can surface the error in the UI."""
    try:
        import requests  # deferred so missing requests gives a clear message
    except ImportError as exc:
        raise RuntimeError(
            "[ToriiGate API] The 'requests' library is not installed. "
            "Run: pip install requests"
        ) from exc

    endpoint = server_url.rstrip("/") + "/v1/chat/completions"
    logger.info("[ToriiGate API] POST → %s  (model=%s)", endpoint, payload.get("model", "?"))

    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"[ToriiGate API] Cannot connect to llama-server at '{server_url}'. "
            "Make sure llama-server is running and the URL is correct.\n"
            f"Detail: {exc}"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"[ToriiGate API] Request timed out after {timeout}s. "
            "Try increasing the timeout or reducing max_tokens.\n"
            f"Detail: {exc}"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"[ToriiGate API] HTTP error: {exc}") from exc

    if not response.ok:
        try:
            error_body = response.json()
        except Exception:
            error_body = response.text
        raise RuntimeError(
            f"[ToriiGate API] Server returned HTTP {response.status_code}.\n"
            f"Response: {json.dumps(error_body, ensure_ascii=False, indent=2)}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"[ToriiGate API] Server returned non-JSON response:\n{response.text[:500]}"
        ) from exc

    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"[ToriiGate API] Unexpected response format:\n"
            f"{json.dumps(data, ensure_ascii=False, indent=2)[:500]}"
        ) from exc

    logger.info("[ToriiGate API] Received %d characters of generated text.", len(text))
    return text


# ---------------------------------------------------------------------------
# All GGUF quantization variants available at:
# https://huggingface.co/DraconicDragon/ToriiGate-0.5-GGUF
# Format used by llama-server router: "repo:quant_tag"
# ---------------------------------------------------------------------------

GGUF_MODEL_NAMES = [
    "DraconicDragon/ToriiGate-0.5-GGUF:Q4_K_M",   # 3.07 GB  ← recommended balance
    "DraconicDragon/ToriiGate-0.5-GGUF:Q4_K_S",   # 2.92 GB
    "DraconicDragon/ToriiGate-0.5-GGUF:Q4_0",     # 2.90 GB
    "DraconicDragon/ToriiGate-0.5-GGUF:Q4_1",     # 3.16 GB
    "DraconicDragon/ToriiGate-0.5-GGUF:IQ4_NL",   # 2.98 GB
    "DraconicDragon/ToriiGate-0.5-GGUF:Q5_K_M",   # 3.51 GB
    "DraconicDragon/ToriiGate-0.5-GGUF:Q5_K_S",   # 3.43 GB
    "DraconicDragon/ToriiGate-0.5-GGUF:Q6_K",     # 3.99 GB
    "DraconicDragon/ToriiGate-0.5-GGUF:Q8_0",     # 5.16 GB
    "DraconicDragon/ToriiGate-0.5-GGUF:Q3_K_L",   # 2.69 GB
    "DraconicDragon/ToriiGate-0.5-GGUF:Q3_K_M",   # 2.54 GB
    "DraconicDragon/ToriiGate-0.5-GGUF:Q3_K_S",   # 2.34 GB
    "DraconicDragon/ToriiGate-0.5-GGUF:Q2_K",     # 2.12 GB
    "DraconicDragon/ToriiGate-0.5-GGUF:bf16",     # 9.70 GB
]



class LlamaCppVisionGenerate:
    """Generate a caption from an image + prompt through local GGUF inference
    or an external llama-server vision endpoint.

    The image is converted from a ComfyUI tensor to a base64-encoded PNG and
    embedded in an OpenAI-compatible multimodal chat-completions request.
    No Transformers or PyTorch model loading is performed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "ComfyUI image tensor (B, H, W, C float32). "
                            "Only the first image in the batch is sent."
                        )
                    },
                ),
                "runtime": (
                    ["local_gguf", "api_server"],
                    {
                        "default": "local_gguf",
                        "tooltip": (
                            "local_gguf runs the GGUF model directly in ComfyUI through "
                            "llama-cpp-python. api_server keeps the old behavior and sends "
                            "requests to an external llama-server."
                        ),
                    },
                ),
                "local_model_path": (
                    _list_llm_gguf_paths(mmproj=False),
                    {
                        "default": DEFAULT_LOCAL_MODEL_PATH,
                        "tooltip": (
                            "Only used when runtime is local_gguf. Auto-scanned from "
                            "ComfyUI/models/LLM, excluding files with mmproj in the name."
                        ),
                    },
                ),
                "local_mmproj_path": (
                    _list_llm_gguf_paths(mmproj=True),
                    {
                        "default": DEFAULT_LOCAL_MMPROJ_PATH,
                        "tooltip": (
                            "Only used when runtime is local_gguf. Auto-scanned from "
                            "ComfyUI/models/LLM, including files with mmproj in the name."
                        ),
                    },
                ),
                "server_url": (
                    "STRING",
                    {
                        "default": "http://127.0.0.1:8080",
                        "tooltip": (
                            "Only used when runtime is api_server. Base URL of the llama-server instance. "
                            "Example: http://127.0.0.1:8080"
                        ),
                    },
                ),
                "model_name": (
                    GGUF_MODEL_NAMES,
                    {
                        "default": "DraconicDragon/ToriiGate-0.5-GGUF:Q4_K_M",
                        "tooltip": (
                            "Only used when runtime is api_server. GGUF quantization to use. The identifier must match what the "
                            "llama-server router registered (shown at startup as 'operator(): ...'). "
                            "Q4_K_M is the recommended balance of quality vs size (3.07 GB). "
                            "Use custom_model_name below to override with any arbitrary string."
                        ),
                    },
                ),
                "timeout": (
                    "FLOAT",
                    {
                        "default": 120.0,
                        "min": 5.0,
                        "max": 600.0,
                        "step": 5.0,
                        "tooltip": (
                            "HTTP request timeout in seconds. "
                            "Increase for slow hardware or very long generations."
                        ),
                    },
                ),
            },
            "optional": {
                "chat_handler": (
                    ["auto", "qwen35-vl", "qwen2.5-vl", "llava-1.5"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "Only used when runtime is local_gguf. auto tries the available "
                            "Qwen3/Qwen3.5 vision handlers first, then older compatible handlers."
                        ),
                    },
                ),
                "n_ctx": (
                    "INT",
                    {
                        "default": 8192,
                        "min": 1024,
                        "max": 131072,
                        "step": 1024,
                        "tooltip": "Only used when runtime is local_gguf. llama.cpp context size.",
                    },
                ),
                "n_gpu_layers": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 999,
                        "step": 1,
                        "tooltip": "Only used when runtime is local_gguf. -1 asks llama.cpp to offload all possible layers.",
                    },
                ),
                "n_threads": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 256,
                        "step": 1,
                        "tooltip": "Only used when runtime is local_gguf. 0 lets llama.cpp choose the thread count.",
                    },
                ),
                "keep_model_alive": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Only used when runtime is local_gguf. Keep the GGUF model cached after generation.",
                    },
                ),
                "verbose": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Only used when runtime is local_gguf. Enables verbose llama.cpp logging.",
                    },
                ),
                "custom_model_name": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Override the model identifier with any custom string. "
                            "Useful when running a non-GGUF backend or a locally "
                            "renamed model. Leave blank to use the dropdown above."
                        ),
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Optional prompt. You can connect the text output from the ToriiGate Grounding Builder here, or type your own.",
                        "forceInput": True,
                    },
                ),
                "max_pixels_mp": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 8.0,
                        "step": 0.1,
                        "tooltip": "Resolution limit sent to the model, in megapixels. Lower values drastically reduce prompt evaluation time (Time To First Token) in llama.cpp.",
                    },
                ),
                "max_new_tokens": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 8192,
                        "step": 64,
                        "tooltip": "Maximum generated tokens.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Generation randomness. 0 is deterministic.",
                    },
                ),
                "decoding": (
                    ["sample", "greedy_fast"],
                    {
                        "default": "sample",
                        "tooltip": "sample uses temperature-based sampling; greedy_fast sets temperature to 0.0.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFF,
                        "step": 1,
                        "tooltip": "Seed for reproducibility. Use 0 for a random seed.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "generate"
    CATEGORY = "ToriiGate/API"

    def generate(
        self,
        image,
        runtime="local_gguf",
        server_url="http://127.0.0.1:8080",
        model_name="DraconicDragon/ToriiGate-0.5-GGUF:Q4_K_M",
        timeout=120.0,
        local_model_path=DEFAULT_LOCAL_MODEL_PATH,
        local_mmproj_path=DEFAULT_LOCAL_MMPROJ_PATH,
        chat_handler="auto",
        n_ctx=8192,
        n_gpu_layers=-1,
        n_threads=0,
        keep_model_alive=True,
        verbose=False,
        custom_model_name="",
        prompt="",
        max_pixels_mp=1.0,
        max_new_tokens=512,
        temperature=0.5,
        decoding="sample",
        seed=0,
    ):
        resolved_model = custom_model_name.strip() if custom_model_name.strip() else model_name.strip()
        actual_server_url = server_url.strip().rstrip("/")
        actual_runtime = (runtime or "local_gguf").strip()

        timeout = _clean_float(timeout, 120.0, 5.0, 600.0)
        max_pixels_mp = _clean_float(max_pixels_mp, 1.0, 0.1, 8.0)
        max_new_tokens = _clean_int(max_new_tokens, 512, 64, 8192)
        n_ctx = _clean_int(n_ctx, 8192, 1024, 131072)
        n_gpu_layers = _clean_int(n_gpu_layers, -1, -1, 999)
        n_threads = _clean_int(n_threads, 0, 0, 256)
        seed = _clean_int(seed, 0, 0, 0xFFFFFFFF)
        actual_temperature = 0.0 if decoding == "greedy_fast" else _clean_float(temperature, 0.5, 0.0, 2.0)

        print(
            f"[ToriiGate] Caption generation started (Vision {actual_runtime}) "
            f"(model={resolved_model if actual_runtime == 'api_server' else local_model_path}, max_pixels={max_pixels_mp}MP, "
            f"decoding={decoding}, temperature={actual_temperature:.2f})."
        )

        if not prompt:
            prompt = "Describe this image in detail."

        # Convert image tensor → base64 PNG
        image_b64 = image_tensor_to_base64(image, float(max_pixels_mp))

        payload = build_vision_payload(
            model_name=resolved_model,
            system_prompt=system_prompt,
            user_text=prompt,
            image_b64=image_b64,
            temperature=actual_temperature,
            max_tokens=int(max_new_tokens),
            seed=int(seed) if seed != 0 else None,
        )

        import time
        start_time = time.perf_counter()
        if actual_runtime == "local_gguf":
            result = run_local_chat_completion(
                messages=payload["messages"],
                model_path=local_model_path,
                mmproj_path=local_mmproj_path,
                chat_handler=chat_handler,
                temperature=actual_temperature,
                max_tokens=int(max_new_tokens),
                seed=int(seed),
                n_ctx=int(n_ctx),
                n_gpu_layers=int(n_gpu_layers),
                n_threads=int(n_threads),
                keep_model_alive=bool(keep_model_alive),
                verbose=bool(verbose),
            )
        elif actual_runtime == "api_server":
            result = send_chat_request(server_url=actual_server_url, payload=payload, timeout=float(timeout))
        else:
            raise RuntimeError(f"[ToriiGate] Unknown runtime: {runtime}")
        elapsed = time.perf_counter() - start_time
        
        print(f"[ToriiGate] Caption generation finished in {elapsed:.1f}s ({len(result)} chars).")
        return (result.strip(),)


# ---------------------------------------------------------------------------
# Node: LlamaCppTextGenerate
# ---------------------------------------------------------------------------

_DEFAULT_TEXT_PROMPT = "Describe the following topic in detail:"


class LlamaCppTextGenerate:
    """Generate a text response through local GGUF inference or an external
    llama-server. No image is sent; no Transformers or PyTorch code is used.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": _DEFAULT_TEXT_PROMPT,
                        "tooltip": "User prompt sent to the llama-server.",
                    },
                ),
                "runtime": (
                    ["local_gguf", "api_server"],
                    {
                        "default": "local_gguf",
                        "tooltip": (
                            "local_gguf runs the GGUF model directly in ComfyUI through "
                            "llama-cpp-python. api_server keeps the old behavior and sends "
                            "requests to an external llama-server."
                        ),
                    },
                ),
                "local_model_path": (
                    _list_llm_gguf_paths(mmproj=False),
                    {
                        "default": DEFAULT_LOCAL_MODEL_PATH,
                        "tooltip": (
                            "Only used when runtime is local_gguf. Auto-scanned from "
                            "ComfyUI/models/LLM, excluding files with mmproj in the name."
                        ),
                    },
                ),
                "server_url": (
                    "STRING",
                    {
                        "default": "http://127.0.0.1:8080",
                        "tooltip": (
                            "Only used when runtime is api_server. Base URL of the llama-server instance. "
                            "Example: http://127.0.0.1:8080"
                        ),
                    },
                ),
                "model_name": (
                    GGUF_MODEL_NAMES,
                    {
                        "default": "DraconicDragon/ToriiGate-0.5-GGUF:Q4_K_M",
                        "tooltip": (
                            "Only used when runtime is api_server. GGUF quantization to use. The identifier must match what the "
                            "llama-server router registered (shown at startup as 'operator(): ...'). "
                            "Q4_K_M is the recommended balance of quality vs size (3.07 GB). "
                            "Use custom_model_name below to override with any arbitrary string."
                        ),
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": (
                            "Sampling temperature. 0 is deterministic (greedy); "
                            "higher values introduce more randomness."
                        ),
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 512,
                        "min": 16,
                        "max": 8192,
                        "step": 64,
                        "tooltip": "Maximum number of tokens to generate.",
                    },
                ),
                "timeout": (
                    "FLOAT",
                    {
                        "default": 120.0,
                        "min": 5.0,
                        "max": 600.0,
                        "step": 5.0,
                        "tooltip": (
                            "HTTP request timeout in seconds. "
                            "Increase for slow hardware or very long generations."
                        ),
                    },
                ),
            },
            "optional": {
                "n_ctx": (
                    "INT",
                    {
                        "default": 8192,
                        "min": 1024,
                        "max": 131072,
                        "step": 1024,
                        "tooltip": "Only used when runtime is local_gguf. llama.cpp context size.",
                    },
                ),
                "n_gpu_layers": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 999,
                        "step": 1,
                        "tooltip": "Only used when runtime is local_gguf. -1 asks llama.cpp to offload all possible layers.",
                    },
                ),
                "n_threads": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 256,
                        "step": 1,
                        "tooltip": "Only used when runtime is local_gguf. 0 lets llama.cpp choose the thread count.",
                    },
                ),
                "keep_model_alive": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Only used when runtime is local_gguf. Keep the GGUF model cached after generation.",
                    },
                ),
                "verbose": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Only used when runtime is local_gguf. Enables verbose llama.cpp logging.",
                    },
                ),
                "custom_model_name": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Override the model identifier with any custom string. "
                            "Useful when running a non-GGUF backend or a locally "
                            "renamed model. Leave blank to use the dropdown above."
                        ),
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Optional system prompt. Leave blank to omit the system turn."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "ToriiGate/API"

    def generate(
        self,
        prompt: str,
        runtime: str = "local_gguf",
        server_url: str = "http://127.0.0.1:8080",
        model_name: str = "DraconicDragon/ToriiGate-0.5-GGUF:Q4_K_M",
        temperature: float = 0.7,
        max_tokens: int = 512,
        timeout: float = 120.0,
        local_model_path: str = DEFAULT_LOCAL_MODEL_PATH,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        n_threads: int = 0,
        keep_model_alive: bool = True,
        verbose: bool = False,
        custom_model_name: str = "",
        system_prompt: str = "",
    ):
        resolved_model = custom_model_name.strip() if custom_model_name.strip() else model_name.strip()
        actual_server_url = server_url.strip().rstrip("/")
        actual_runtime = (runtime or "local_gguf").strip()
        timeout = _clean_float(timeout, 120.0, 5.0, 600.0)
        max_tokens = _clean_int(max_tokens, 512, 16, 8192)
        n_ctx = _clean_int(n_ctx, 8192, 1024, 131072)
        n_gpu_layers = _clean_int(n_gpu_layers, -1, -1, 999)
        n_threads = _clean_int(n_threads, 0, 0, 256)
        actual_temperature = _clean_float(temperature, 0.7, 0.0, 2.0)

        print(
            f"[ToriiGate] Caption generation started (Text {actual_runtime}) "
            f"(model={resolved_model if actual_runtime == 'api_server' else local_model_path}, "
            f"max_tokens={max_tokens}, temperature={actual_temperature:.2f})."
        )

        payload = build_text_payload(
            model_name=resolved_model,
            system_prompt=system_prompt,
            user_text=prompt,
            temperature=actual_temperature,
            max_tokens=int(max_tokens),
        )

        import time
        start_time = time.perf_counter()
        if actual_runtime == "local_gguf":
            result = run_local_chat_completion(
                messages=payload["messages"],
                model_path=local_model_path,
                mmproj_path=None,
                chat_handler="",
                temperature=actual_temperature,
                max_tokens=int(max_tokens),
                seed=0,
                n_ctx=int(n_ctx),
                n_gpu_layers=int(n_gpu_layers),
                n_threads=int(n_threads),
                keep_model_alive=bool(keep_model_alive),
                verbose=bool(verbose),
            )
        elif actual_runtime == "api_server":
            result = send_chat_request(server_url=actual_server_url, payload=payload, timeout=float(timeout))
        else:
            raise RuntimeError(f"[ToriiGate] Unknown runtime: {runtime}")
        elapsed = time.perf_counter() - start_time
        
        print(f"[ToriiGate] Caption generation finished in {elapsed:.1f}s ({len(result)} chars).")
        return (result.strip(),)


# ---------------------------------------------------------------------------
# ComfyUI registration maps (imported by nodes.py)
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS_API = {
    "ToriiGate_LlamaCppVisionGenerate": LlamaCppVisionGenerate,
    "ToriiGate_LlamaCppTextGenerate": LlamaCppTextGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS_API = {
    "ToriiGate_LlamaCppVisionGenerate": "ToriiGate Llama.cpp Vision Generate",
    "ToriiGate_LlamaCppTextGenerate": "ToriiGate Llama.cpp Text Generate",
}
