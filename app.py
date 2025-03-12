import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Import our custom modules
import ui
import calc
import info

# Set page config
st.set_page_config(page_title="LLM Hardware Requirements Calculator", layout="wide")

# Add title and description
st.title("LLM Hardware Requirements Calculator")
st.markdown("""
This application helps you estimate hardware requirements for running and training Large Language Models (LLMs)
based on model size, quantization settings, operation mode, and data sensitivity.
""")

# Generate the UI components and get user inputs
model_config = ui.create_sidebar_inputs()

# Calculate requirements based on user inputs
if model_config["operation_mode"] == "Inference":
    memory_data = calc.calculate_vram_usage(
        model_config["num_params"], 
        model_config["quantization"], 
        model_config["context_length"], 
        model_config["cuda_support"], 
        model_config["operation_mode"]
    )
else:  # Training mode
    memory_data = calc.calculate_vram_usage(
        model_config["num_params"], 
        model_config["quantization"], 
        model_config["context_length"], 
        model_config["cuda_support"], 
        model_config["operation_mode"],
        model_config.get("batch_size", 1),
        model_config.get("gradient_accumulation", 1),
        model_config.get("precision_type", "FP16"),
        model_config.get("optimizer_choice", "AdamW")
    )

# Extract memory components
vram_needed = memory_data["total"]
model_weight_memory = memory_data["model_weights"]
kv_cache_memory = memory_data["kv_cache"]
overhead_memory = memory_data["overhead"]

# Training-specific memory components
optimizer_memory = memory_data.get("optimizer", 0)
gradients_memory = memory_data.get("gradients", 0)
activation_memory = memory_data.get("activations", 0)

# Calculate additional requirements
# Calculate additional requirements
disk_size = calc.calculate_disk_size(model_config["num_params"], model_config["quantization"])
system_ram = calc.calculate_system_ram(vram_needed)
gpu_config = calc.recommend_gpu_config(vram_needed, model_config["sensitivity_level"])
num_gpus = calc.calculate_gpus_required(vram_needed, model_config["sensitivity_level"])

# Determine GPU tier based on recommended config
if "great-lakes" in gpu_config.lower():
    if "a40" in gpu_config.lower() or "a100" in gpu_config.lower():
        gpu_tier = "professional"
    else:
        gpu_tier = "high-end"
elif "armis2" in gpu_config.lower():
    if "geforce" in gpu_config.lower():
        gpu_tier = "consumer"
    else:
        gpu_tier = "professional"
elif "lighthouse" in gpu_config.lower():
    gpu_tier = "professional"
else:
    # Default if no matching partition found
    gpu_tier = "professional"

# Calculate performance metrics
if model_config["operation_mode"] == "Inference":
    tokens_per_second = calc.estimate_tokens_per_second(
        model_config["num_params"],
        model_config["quantization"],
        model_config["context_length"],
        model_config["cuda_support"],
        gpu_tier
    )
    tokens_per_second_training = None
else:
    tokens_per_second = None
    # Calculate approximate training time per epoch
    if model_config["cuda_support"]:
        if vram_needed <= 24:  # Can fit on a single high-end GPU
            tokens_per_second_training = max(1, calc.estimate_tokens_per_second(
                model_config["num_params"],
                model_config["quantization"],
                model_config["context_length"],
                model_config["cuda_support"],
                gpu_tier
            ) / 15)  # Training is ~15x slower than inference
        elif vram_needed <= 80:  # Single A100
            tokens_per_second_training = max(1, calc.estimate_tokens_per_second(
                model_config["num_params"],
                model_config["quantization"],
                model_config["context_length"],
                model_config["cuda_support"],
                gpu_tier
            ) / 12)
        else:  # Multi-GPU
            tokens_per_second_training = max(1, calc.estimate_tokens_per_second(
                model_config["num_params"],
                model_config["quantization"],
                model_config["context_length"],
                model_config["cuda_support"],
                gpu_tier
            ) / 10)
    else:
        tokens_per_second_training = max(0.1, calc.estimate_tokens_per_second(
            model_config["num_params"],
            model_config["quantization"],
            model_config["context_length"],
            model_config["cuda_support"],
            gpu_tier
        ) / 50)  # CPU training is extremely slow

# Create the main display panels
ui.display_requirements(
    model_config,
    vram_needed,
    model_weight_memory,
    kv_cache_memory,
    overhead_memory,
    optimizer_memory,
    gradients_memory,
    activation_memory,
    disk_size,
    system_ram,
    gpu_config,
    num_gpus,
    tokens_per_second,
    tokens_per_second_training
)

# Display charts for memory breakdown
ui.display_memory_charts(
    model_config["operation_mode"],
    model_weight_memory,
    kv_cache_memory,
    overhead_memory,
    optimizer_memory,
    gradients_memory,
    activation_memory,
    disk_size,
    system_ram,
    vram_needed
)

# Create hardware comparison with sensitivity filtering
ui.display_hardware_comparison(
    model_config,
    vram_needed,
    calc.filter_hardware_options(model_config["sensitivity_level"])
)

# Display model comparisons with sensitivity awareness
ui.display_model_comparison(
    model_config,
    calc.get_known_models(),
    calc.calculate_vram_usage,
    calc.estimate_tokens_per_second,
    calc.filter_hardware_options
)

# Display reference information
st.markdown("---")
st.markdown(info.get_reference_information())

# Add information about sensitivity levels
st.markdown("---")
st.subheader("Data Sensitivity Information")
st.markdown("""
### Understanding Sensitivity Levels

- **Moderate Sensitivity**: Use this for standard research data that doesn't contain sensitive information. Only great-lakes partitions are available.
- **High Sensitivity**: Use this for projects that can access special-purpose hardware. All partitions (great-lakes, armis2, lighthouse) are available.

Always ensure your data handling complies with institutional policies and applicable regulations.
""")