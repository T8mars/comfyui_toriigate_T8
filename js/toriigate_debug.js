import { app } from "../../scripts/app.js";

const NUMERIC_WIDGETS = {
    timeout: { default: 120, min: 5, max: 600, integer: false },
    n_ctx: { default: 8192, min: 1024, max: 131072, integer: true },
    n_gpu_layers: { default: -1, min: -1, max: 999, integer: true },
    n_threads: { default: 0, min: 0, max: 256, integer: true },
    max_pixels_mp: { default: 1, min: 0.1, max: 8, integer: false },
    max_new_tokens: { default: 512, min: 64, max: 8192, integer: true },
    max_tokens: { default: 512, min: 16, max: 8192, integer: true },
    temperature: { default: 0.5, min: 0, max: 2, integer: false },
    seed: { default: 0, min: 0, max: 0xffffffff, integer: true },
};

const COMBO_WIDGETS = {
    runtime: { default: "local_gguf", values: ["local_gguf", "api_server"] },
    chat_handler: { default: "auto", values: ["auto", "qwen35-vl", "qwen2.5-vl", "llava-1.5"] },
    decoding: { default: "sample", values: ["sample", "greedy_fast"] },
};

const STRING_WIDGETS = {
    server_url: "http://127.0.0.1:8080",
    custom_model_name: "",
    prompt: "",
    system_prompt: "",
};

const BOOLEAN_WIDGETS = {
    keep_model_alive: true,
    verbose: false,
    show_generation_progress: false,
};

const VISION_LEGACY_WIDGETS = [
    "server_url",
    "model_name",
    "timeout",
    "custom_model_name",
    "prompt",
    "max_pixels_mp",
    "max_new_tokens",
    "temperature",
    "decoding",
    "seed",
];

const TARGET_NODES = new Set([
    "ToriiGate_LlamaCppVisionGenerate",
    "ToriiGate_LlamaCppTextGenerate",
    "ToriiGate_Captioner",
]);

function normalizeNumber(value, config) {
    let number = Number(value);
    if (!Number.isFinite(number)) {
        number = config.default;
    }

    number = Math.min(config.max, Math.max(config.min, number));
    if (config.integer) {
        number = Math.round(number);
    }
    return number;
}

function getWidgetConfig(nodeName, widgetName) {
    const config = NUMERIC_WIDGETS[widgetName];
    if (!config) {
        return null;
    }

    if (nodeName === "ToriiGate_LlamaCppTextGenerate" && widgetName === "temperature") {
        return { ...config, default: 0.7 };
    }

    if (nodeName === "ToriiGate_Captioner" && widgetName === "max_new_tokens") {
        return { ...config, max: 4096 };
    }

    return config;
}

function getWidgetValues(widget) {
    const values = widget?.options?.values ?? widget?.options?.items;
    return Array.isArray(values) ? values : null;
}

function setWidgetValue(node, name, value) {
    const widget = node?.widgets?.find((candidate) => candidate.name === name);
    if (widget) {
        widget.value = value;
    }
}

function isValidComboValue(widget, value, fallbackConfig) {
    const values = getWidgetValues(widget) ?? fallbackConfig?.values;
    return Array.isArray(values) && values.includes(value);
}

function normalizeCombo(widget, fallbackConfig) {
    if (isValidComboValue(widget, widget.value, fallbackConfig)) {
        return widget.value;
    }

    const values = getWidgetValues(widget) ?? fallbackConfig?.values ?? [];
    if (fallbackConfig?.default && values.includes(fallbackConfig.default)) {
        return fallbackConfig.default;
    }

    return values[0] ?? fallbackConfig?.default ?? widget.value;
}

function sanitizeComboWidgets(node) {
    if (!node?.widgets) {
        return false;
    }

    let changed = false;
    for (const widget of node.widgets) {
        const fallbackConfig = COMBO_WIDGETS[widget.name];
        const values = getWidgetValues(widget) ?? fallbackConfig?.values;
        if (!Array.isArray(values)) {
            continue;
        }

        const normalized = normalizeCombo(widget, fallbackConfig);
        if (!Object.is(widget.value, normalized)) {
            widget.value = normalized;
            changed = true;
        }
    }
    return changed;
}

function sanitizeStringWidgets(node) {
    if (!node?.widgets) {
        return false;
    }

    let changed = false;
    for (const widget of node.widgets) {
        if (!(widget.name in STRING_WIDGETS)) {
            continue;
        }

        if (typeof widget.value !== "string") {
            widget.value = STRING_WIDGETS[widget.name];
            changed = true;
        }
    }
    return changed;
}

function sanitizeBooleanWidgets(node) {
    if (!node?.widgets) {
        return false;
    }

    let changed = false;
    for (const widget of node.widgets) {
        if (!(widget.name in BOOLEAN_WIDGETS)) {
            continue;
        }

        if (typeof widget.value !== "boolean") {
            widget.value = Boolean(widget.value);
            changed = true;
        }
    }
    return changed;
}

function migrateLegacyVisionValues(node, serialized) {
    const values = serialized?.widgets_values;
    if (!Array.isArray(values) || values.length < VISION_LEGACY_WIDGETS.length) {
        return false;
    }

    const runtimeWidget = node?.widgets?.find((widget) => widget.name === "runtime");
    const decodingValue = values[VISION_LEGACY_WIDGETS.indexOf("decoding")];
    const looksLikeLegacyApiNode =
        typeof values[0] === "string" &&
        /^https?:\/\//.test(values[0]) &&
        !isValidComboValue(runtimeWidget, values[0], COMBO_WIDGETS.runtime) &&
        COMBO_WIDGETS.decoding.values.includes(decodingValue);

    if (!looksLikeLegacyApiNode) {
        return false;
    }

    setWidgetValue(node, "runtime", "api_server");
    for (let i = 0; i < VISION_LEGACY_WIDGETS.length; i++) {
        setWidgetValue(node, VISION_LEGACY_WIDGETS[i], values[i]);
    }
    return true;
}

function sanitizeNumericWidgets(node, nodeName) {
    if (!node?.widgets) {
        return false;
    }

    let changed = false;
    for (const widget of node.widgets) {
        const config = getWidgetConfig(nodeName, widget.name);
        if (!config) {
            continue;
        }

        const normalized = normalizeNumber(widget.value, config);
        if (!Object.is(widget.value, normalized)) {
            widget.value = normalized;
            changed = true;
        }
    }
    return changed;
}

function sanitizeAllWidgets(node, nodeName) {
    const numericChanged = sanitizeNumericWidgets(node, nodeName);
    const comboChanged = sanitizeComboWidgets(node);
    const stringChanged = sanitizeStringWidgets(node);
    const booleanChanged = sanitizeBooleanWidgets(node);
    return numericChanged || comboChanged || stringChanged || booleanChanged;
}

function getNodeName(node) {
    return node?.__toriigateNodeName ?? node?.comfyClass ?? node?.type;
}

function sanitizeGraphNodes() {
    let changed = false;
    for (const node of app.graph?._nodes ?? []) {
        const nodeName = getNodeName(node);
        if (!TARGET_NODES.has(nodeName)) {
            continue;
        }

        changed = sanitizeAllWidgets(node, nodeName) || changed;
    }

    if (changed) {
        app.graph?.setDirtyCanvas(true, true);
    }
}

function installNumericSanitizer(nodeType, nodeName) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        if (onNodeCreated) {
            onNodeCreated.apply(this, arguments);
        }

        const node = this;
        node.__toriigateNodeName = nodeName;
        const sanitize = () => {
            if (sanitizeAllWidgets(node, nodeName)) {
                app.graph?.setDirtyCanvas(true, true);
            }
        };

        for (const widget of node.widgets ?? []) {
            const shouldWrap =
                getWidgetConfig(nodeName, widget.name) ||
                COMBO_WIDGETS[widget.name] ||
                widget.options?.values ||
                widget.options?.items ||
                widget.name in STRING_WIDGETS ||
                widget.name in BOOLEAN_WIDGETS;
            if (!shouldWrap) {
                continue;
            }

            const originalCallback = widget.callback;
            widget.callback = function () {
                const result = originalCallback ? originalCallback.apply(this, arguments) : undefined;
                sanitize();
                return result;
            };
        }

        const onConfigure = node.onConfigure;
        node.onConfigure = function () {
            if (onConfigure) {
                onConfigure.apply(this, arguments);
            }
            if (nodeName === "ToriiGate_LlamaCppVisionGenerate") {
                migrateLegacyVisionValues(node, arguments[0]);
            }
            sanitize();
        };

        const onAdded = node.onAdded;
        node.onAdded = function () {
            if (onAdded) {
                onAdded.apply(this, arguments);
            }
            sanitize();
        };

        const onDrawForeground = node.onDrawForeground;
        node.onDrawForeground = function () {
            if (onDrawForeground) {
                onDrawForeground.apply(this, arguments);
            }
            sanitizeAllWidgets(node, nodeName);
        };

        const onSerialize = node.onSerialize;
        node.onSerialize = function () {
            sanitize();
            if (onSerialize) {
                return onSerialize.apply(this, arguments);
            }
        };

        setTimeout(sanitize, 0);
        setTimeout(sanitize, 100);
    };
}

app.registerExtension({
    name: "ComfyUI.ToriiGate.NumericWidgetSanitizer",
    async setup() {
        if (app.__toriigateQueueSanitizerInstalled) {
            return;
        }

        const originalQueuePrompt = app.queuePrompt;
        if (typeof originalQueuePrompt === "function") {
            app.queuePrompt = async function () {
                sanitizeGraphNodes();
                return originalQueuePrompt.apply(this, arguments);
            };
            app.__toriigateQueueSanitizerInstalled = true;
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (TARGET_NODES.has(nodeData.name)) {
            installNumericSanitizer(nodeType, nodeData.name);
        }
    },
});
