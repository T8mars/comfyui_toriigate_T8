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

function installNumericSanitizer(nodeType, nodeName) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        if (onNodeCreated) {
            onNodeCreated.apply(this, arguments);
        }

        const node = this;
        const sanitize = () => {
            if (sanitizeNumericWidgets(node, nodeName)) {
                app.graph?.setDirtyCanvas(true, true);
            }
        };

        for (const widget of node.widgets ?? []) {
            if (!getWidgetConfig(nodeName, widget.name)) {
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
            sanitizeNumericWidgets(node, nodeName);
        };

        setTimeout(sanitize, 0);
        setTimeout(sanitize, 100);
    };
}

app.registerExtension({
    name: "ComfyUI.ToriiGate.NumericWidgetSanitizer",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (TARGET_NODES.has(nodeData.name)) {
            installNumericSanitizer(nodeType, nodeData.name);
        }
    },
});
