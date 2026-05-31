import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ComfyUI.ToriiGate.GroundingBuilder",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ToriiGate_GroundingBuilder") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }
                
                const node = this;
                


                
                const updateWidgets = () => {
                    const add_tags = node.widgets.find(w => w.name === "add_tags")?.value;
                    const add_character_list = node.widgets.find(w => w.name === "add_character_list")?.value;
                    const character_count = parseInt(node.widgets.find(w => w.name === "character_count")?.value || 1);
                    const add_character_tags = node.widgets.find(w => w.name === "add_character_tags")?.value;
                    const add_character_descriptions = node.widgets.find(w => w.name === "add_character_descriptions")?.value;
                    
                    for (const w of node.widgets) {
                        if (["add_tags", "add_character_list", "character_count", "add_character_tags", "add_character_descriptions"].includes(w.name)) {
                            continue;
                        }

                        let show = true;
                        
                        if (w.name === "tags") show = add_tags;
                        else if (w.name === "character_names") show = add_character_list;
                        else {
                            for (let i = 1; i <= 5; i++) {
                                if (w.name === `char${i}_name`) show = i <= character_count;
                                else if (w.name === `char${i}_tags`) show = i <= character_count && add_character_tags;
                                else if (w.name === `char${i}_description`) show = i <= character_count && add_character_descriptions;
                            }
                        }
                        
                        if (show) {
                            w.type = (w.name.includes("tags") || w.name.includes("character_names") || w.name.includes("description")) ? "customtext" : "text";
                            delete w.computeSize;
                            w.hidden = false;
                            
                            // Try to show HTML elements
                            if (w.inputEl) { w.inputEl.style.display = ""; w.inputEl.hidden = false; }
                            if (w.element) { w.element.style.display = ""; w.element.hidden = false; }
                        } else {
                            w.type = "hidden";
                            w.computeSize = () => [0, -4];
                            w.hidden = true;
                            
                            // Try to hide HTML elements
                            if (w.inputEl) { w.inputEl.style.display = "none"; w.inputEl.hidden = true; }
                            if (w.element) { w.element.style.display = "none"; w.element.hidden = true; }
                        }
                    }
                    
                    setTimeout(() => {
                        const size = node.computeSize();
                        node.setSize([Math.max(size[0], node.size[0]), Math.max(size[1], 100)]);
                        app.graph.setDirtyCanvas(true, true);
                    }, 10);
                };
                
                // Attach callbacks to toggle widgets
                const toggleNames = [
                    "add_tags", 
                    "add_character_list", 
                    "character_count", 
                    "add_character_tags", 
                    "add_character_descriptions"
                ];
                
                for (const w of node.widgets) {
                    if (toggleNames.includes(w.name)) {
                        const originalCallback = w.callback;
                        w.callback = function() {
                            const ret = originalCallback ? originalCallback.apply(this, arguments) : undefined;
                            updateWidgets();
                            return ret;
                        };
                    }
                }
                
                const onAdded = node.onAdded;
                node.onAdded = function() {
                    if (onAdded) onAdded.apply(this, arguments);
                    updateWidgets();
                };

                const onConfigure = node.onConfigure;
                node.onConfigure = function() {
                    if (onConfigure) onConfigure.apply(this, arguments);
                    updateWidgets();
                };
                
                // Initial updates
                setTimeout(updateWidgets, 50);
                setTimeout(updateWidgets, 200);
            };
        }
    }
});
