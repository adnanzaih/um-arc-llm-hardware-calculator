# Add reference information
st.markdown("---")
st.markdown("""
### Training vs. Inference Mode Explained:
- **Inference**: Running a pre-trained model to generate predictions or text
- **Training**: Updating model weights to learn from data, requiring much more memory
- **Key Differences**:
  - Training requires storing optimizer states, gradients, and full activations
  - Inference primarily needs model weights and KV cache
  - Training memory usage scales with batch size and sequence length

### Batch Size and Gradient Accumulation:
- **Batch Size**: Number of samples processed simultaneously
- **Gradient Accumulation**: Number of forward/backward passes before weight update
- **Memory Impact**: Larger batch sizes require more memory for activations
- **Performance Impact**: Larger effective batch sizes (batch_size × grad_accum) often improve training stability

### KV Cache Explained:
- **What it is**: Storage of previously computed attention keys (K) and values (V) during inference
- **Why it matters**: Prevents recalculating attention for previous tokens when generating new tokens
- **Scaling**: Grows linearly with context length and model layers
- **Formula**: KV Cache Size ≈ 2 * num_layers * seq_length * hidden_size * bytes_per_element
- **Memory impact**: For long contexts (32K+), KV cache can require more memory than the model weights

### Notes on Quantization:
- **Q2-Q8**: Refers to bits per weight in quantized models
- **FP16/FP32**: Full precision floating point (16/32 bits per weight)
- **GPTQ/AWQ**: Advanced quantization methods (typically ~4 bits per weight)

### Context Length Impact:
- **Memory Usage**: Longer contexts require more VRAM for KV cache storage
- **Performance**: Very long contexts can significantly reduce inference speed
- **Tradeoffs**: Higher context enables processing more text but increases resource requirements

### CUDA Support Impact:
- **With CUDA**: Enables GPU acceleration but may require ~10% additional memory for CUDA-specific optimizations
- **Without CUDA**: May use slightly less memory but performance will be significantly slower
- CUDA support is essential for practical inference with larger models

### Tokens Per Second (TPS) Performance:
- **40+ TPS**: Excellent (real-time conversation)
- **20-40 TPS**: Very good performance
- **10-20 TPS**: Good performance
- **5-10 TPS**: Adequate for most use cases
- **1-5 TPS**: Noticeable lag in responses
- **<1 TPS**: Very slow, not suitable for interactive applications

### Hardware Recommendations:
- Models under 10B parameters can run on consumer GPUs with appropriate quantization
- 10-70B models typically require high-end GPUs or multiple GPUs
- 100B+ models generally require multiple professional GPUs or specialized hardware
- Training requires 2-8x more memory than inference for the same model

### References:
- These calculations are approximations and actual requirements may vary
- KV cache calculations assume standard attention mechanisms
- Performance estimates assume standard inference settings
- Actual performance may vary based on specific hardware, software optimizations, and batch sizes
""")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="LLM Hardware Requirements Calculator", layout="wide")

# Add title and description
st.title("LLM Hardware Requirements Calculator")
st.markdown("""
This application helps you estimate hardware requirements for running Large Language Models (LLMs)
based on model size and quantization settings.
""")

# Create sidebar for configuration options
st.sidebar.header("Model Configuration")

# Model size selection - offering both slider and direct input
param_input_method = st.sidebar.radio(
    "Parameter Input Method",
    options=["Slider", "Direct Input"],
    index=0
)

if param_input_method == "Slider":
    num_params = st.sidebar.slider(
        "Number of Parameters (Billions)", 
        min_value=1, 
        max_value=1000, 
        value=7,
        step=1
    )
else:
    num_params = st.sidebar.number_input(
        "Number of Parameters (Billions)",
        min_value=1,
        max_value=10000,  # Increased max for direct input
        value=7,
        step=1
    )

# Quantization selection
quantization_options = ["Q2", "Q3", "Q4", "Q5", "Q7", "Q8", "FP16", "FP32", "GPTQ", "AWQ"]
quantization = st.sidebar.selectbox(
    "Model Quantization", 
    options=quantization_options,
    index=5  # Default to Q8
)

# Context length selection
context_length = st.sidebar.slider(
    "Context Length (tokens)",
    min_value=512,
    max_value=128000,
    value=4096,
    step=512,
    help="Maximum number of tokens the model can process at once"
)

# CUDA support option
cuda_support = st.sidebar.checkbox("CUDA Support", value=True, help="Enable for NVIDIA GPU acceleration")

# Training vs Inference mode
operation_mode = st.sidebar.radio(
    "Operation Mode",
    options=["Inference", "Training"],
    index=0,
    help="Inference requires less memory than training"
)

# Batch size for training
if operation_mode == "Training":
    batch_size = st.sidebar.slider(
        "Batch Size",
        min_value=1,
        max_value=128,
        value=8,
        step=1,
        help="Number of samples processed simultaneously during training"
    )
    
    gradient_accumulation = st.sidebar.slider(
        "Gradient Accumulation Steps",
        min_value=1,
        max_value=64,
        value=4,
        step=1,
        help="Number of forward passes before updating weights; helps with memory constraints"
    )
    
    precision_type = st.sidebar.selectbox(
        "Training Precision",
        options=["FP16", "BF16", "FP32", "8-bit (int8)", "4-bit (int4)"],
        index=0,
        help="Numerical precision used during training"
    )
    
    optimizer_choice = st.sidebar.selectbox(
        "Optimizer",
        options=["AdamW", "Adam", "SGD", "Lion", "Adafactor"],
        index=0,
        help="Optimizer affects memory usage"
    )

# Function to calculate memory usage based on model size and quantization
def calculate_model_weights_memory(params_billions, quant_type):
    """Calculate approximate memory needed for model weights based on parameters and quantization."""
    params = params_billions * 1e9  # Convert to actual number of parameters
    
    # Bits per parameter based on quantization type
    bits_per_param = {
        "Q2": 2,
        "Q3": 3,
        "Q4": 4,
        "Q5": 5,
        "Q7": 7,
        "Q8": 8,
        "FP16": 16,
        "FP32": 32,
        "GPTQ": 4,  # Approximation for GPTQ
        "AWQ": 4    # Approximation for AWQ
    }
    
    # Calculate bytes needed for weights
    bytes_needed = (params * bits_per_param[quant_type]) / 8
    
    # Convert to GB
    gb_needed = bytes_needed / 1e9
    
    return gb_needed

def calculate_kv_cache_size(params_billions, context_length):
    """
    Calculate the memory needed for KV cache based on model size and context length.
    
    KV cache size = 2 * num_layers * seq_length * hidden_size * bytes_per_element
    """
    # Estimate model architecture based on parameter count
    # These are approximations based on common architectures
    if params_billions <= 1:
        num_layers = 12
        hidden_size = 768
    elif params_billions <= 3:
        num_layers = 24
        hidden_size = 1024
    elif params_billions <= 7:
        num_layers = 32
        hidden_size = 4096
    elif params_billions <= 13:
        num_layers = 40
        hidden_size = 5120
    elif params_billions <= 30:
        num_layers = 60
        hidden_size = 6144
    elif params_billions <= 70:
        num_layers = 80
        hidden_size = 8192
    elif params_billions <= 100:
        num_layers = 96
        hidden_size = 8192
    else:  # Very large models
        num_layers = 128
        hidden_size = 12288
    
    # KV cache is typically stored in FP16 (2 bytes per element)
    bytes_per_element = 2
    
    # Calculate KV cache size (keys + values for each layer)
    kv_cache_bytes = 2 * num_layers * context_length * hidden_size * bytes_per_element
    
    # Return size in GB
    return kv_cache_bytes / 1e9

def calculate_optimizer_states_memory(params_billions, optimizer_type="AdamW", precision="FP16"):
    """Calculate memory needed for optimizer states during training."""
    params = params_billions * 1e9
    
    # Get bytes per parameter based on precision
    precision_bytes = {
        "FP32": 4,
        "FP16": 2,
        "BF16": 2,
        "8-bit (int8)": 1,
        "4-bit (int4)": 0.5
    }
    
    bytes_per_param = precision_bytes.get(precision, 2)  # Default to FP16
    
    # Calculate memory needed for optimizer states
    # Different optimizers have different memory requirements
    if optimizer_type == "AdamW" or optimizer_type == "Adam":
        # Adam/AdamW stores 2 additional states (momentum and variance) per parameter
        optimizer_multiplier = 2
    elif optimizer_type == "Adafactor":
        # Adafactor uses factorized states to reduce memory
        optimizer_multiplier = 1.5
    elif optimizer_type == "Lion":
        # Lion uses 1 additional state per parameter
        optimizer_multiplier = 1
    else:  # SGD
        # SGD with momentum uses 1 additional state per parameter
        optimizer_multiplier = 1
    
    # Calculate total bytes for optimizer states
    optimizer_bytes = params * bytes_per_param * optimizer_multiplier
    
    # Convert to GB
    return optimizer_bytes / 1e9

def calculate_gradients_memory(params_billions, precision="FP16"):
    """Calculate memory needed for gradients during training."""
    params = params_billions * 1e9
    
    # Get bytes per parameter based on precision
    precision_bytes = {
        "FP32": 4,
        "FP16": 2,
        "BF16": 2,
        "8-bit (int8)": 1,
        "4-bit (int4)": 0.5
    }
    
    bytes_per_param = precision_bytes.get(precision, 2)  # Default to FP16
    
    # Calculate memory for gradients - each parameter needs gradient storage
    gradient_bytes = params * bytes_per_param
    
    # Convert to GB
    return gradient_bytes / 1e9

def calculate_activation_memory(params_billions, batch_size, context_length, precision="FP16"):
    """Estimate memory needed for activations during training."""
    # This is a rough estimate based on model architecture and batch size
    # Activations depend on sequence length, model architecture, and batch size
    
    # For a transformer, activations are roughly proportional to:
    # batch_size * context_length * hidden_size * num_layers
    
    # Estimate hidden size and layers based on parameter count
    if params_billions <= 1:
        hidden_size = 768
        num_layers = 12
    elif params_billions <= 7:
        hidden_size = 4096
        num_layers = 32
    elif params_billions <= 13:
        hidden_size = 5120
        num_layers = 40
    elif params_billions <= 70:
        hidden_size = 8192
        num_layers = 80
    else:
        hidden_size = 12288
        num_layers = 96
    
    # Get bytes per activation based on precision
    precision_bytes = {
        "FP32": 4,
        "FP16": 2,
        "BF16": 2,
        "8-bit (int8)": 1,
        "4-bit (int4)": 0.5
    }
    
    bytes_per_activation = precision_bytes.get(precision, 2)
    
    # Factor to account for different activation sizes in different parts of the network
    # This is a simplified approximation
    activation_factor = 4
    
    # Calculate activation memory
    activation_bytes = batch_size * context_length * hidden_size * num_layers * activation_factor * bytes_per_activation
    
    # Convert to GB
    return activation_bytes / 1e9
    
def calculate_vram_usage(params_billions, quant_type, context_length, has_cuda=True,
                         operation_mode="Inference", batch_size=1, grad_accum=1,
                         precision="FP16", optimizer="AdamW"):
    """Calculate approximate total VRAM needed including model weights and KV cache."""
    # Calculate model weights memory
    model_memory = calculate_model_weights_memory(params_billions, quant_type)
    
    # Calculate memory differently based on operation mode
    if operation_mode == "Inference":
        # Inference mode - we need model weights and KV cache
        kv_cache_memory = calculate_kv_cache_size(params_billions, context_length)
        optimizer_memory = 0
        gradients_memory = 0
        activation_memory = 0
        
        # Small overhead for other runtime requirements (approximately 5%)
        other_overhead = model_memory * 0.05
        
    else:  # Training mode
        # For training, we need space for optimizer states, gradients, and activations
        kv_cache_memory = calculate_kv_cache_size(params_billions, context_length * batch_size / grad_accum)
        
        # Optimizer states memory
        optimizer_memory = calculate_optimizer_states_memory(params_billions, optimizer, precision)
        
        # Gradients memory
        gradients_memory = calculate_gradients_memory(params_billions, precision)
        
        # Activation memory (depends on batch size)
        activation_memory = calculate_activation_memory(params_billions, batch_size / grad_accum, context_length, precision)
        
        # Training has more overhead
        other_overhead = model_memory * 0.1
    
    # Add up all memory components
    total_memory = model_memory + kv_cache_memory + optimizer_memory + gradients_memory + activation_memory + other_overhead
    
    # CUDA optimization factor
    if has_cuda:
        # CUDA typically requires extra memory for optimization structures
        cuda_factor = 1.1  # 10% extra for CUDA-specific allocations
    else:
        # CPU-only or other backends might use slightly less memory
        cuda_factor = 0.95  # 5% less without CUDA optimizations
    
    total_memory = total_memory * cuda_factor
    
    return {
        "total": total_memory,
        "model_weights": model_memory,
        "kv_cache": kv_cache_memory,
        "optimizer": optimizer_memory,
        "gradients": gradients_memory,
        "activations": activation_memory,
        "overhead": other_overhead
    }

def calculate_disk_size(params_billions, quant_type):
    """Calculate approximate disk space needed for model storage."""
    # For disk storage, we only need to account for the model weights
    # KV cache is only used at runtime
    return calculate_model_weights_memory(params_billions, quant_type)

def calculate_system_ram(vram_total):
    """Estimate system RAM requirements based on VRAM needs."""
    # System RAM should be at least 2x VRAM for efficient operation
    return max(16, vram_total * 2)  # Minimum 16GB

def recommend_gpu_config(vram_needed):
    """Recommend GPU configuration based on VRAM requirements."""
    if vram_needed <= 8:
        return "Consumer GPU (RTX 3060 or higher)"
    elif vram_needed <= 24:
        return "High-end consumer GPU (RTX 4090 or similar)"
    elif vram_needed <= 48:
        return "Professional GPU (RTX A6000, A100)"
    else:
        return "Multiple professional GPUs or cloud solutions"

def calculate_gpus_required(vram_needed):
    """Calculate number of GPUs needed based on VRAM requirements."""
    # Assuming high-end GPUs with ~80GB VRAM (A100)
    gpus_needed = np.ceil(vram_needed / 80)
    return max(1, int(gpus_needed))

    # Function to estimate tokens per second based on model and hardware
def estimate_tokens_per_second(params_billions, quant_type, context_length, has_cuda=True, gpu_tier="consumer"):
        """
        Estimate tokens per second for inference based on model size, quantization, and hardware.
        
        Args:
            params_billions: Model size in billions of parameters
            quant_type: Quantization method
            context_length: Context window size in tokens
            has_cuda: Whether CUDA is enabled
            gpu_tier: "consumer", "high-end", "professional", or "multi-gpu"
        
        Returns:
            Estimated tokens per second for inference
        """
        # Base token generation speed based on model size (approximate values)
        # These are empirical estimates from various benchmarks and real-world usage
        if params_billions <= 1:
            base_tps = 100  # Small models are very fast
        elif params_billions <= 7:
            base_tps = 70   # 7B models like Llama 2 (7B)
        elif params_billions <= 13:
            base_tps = 40   # 13B models like Llama 2 (13B)
        elif params_billions <= 30:
            base_tps = 25   # Medium models
        elif params_billions <= 70:
            base_tps = 15   # 70B models like Llama 2 (70B)
        elif params_billions <= 100:
            base_tps = 8    # Large models
        else:
            base_tps = 4    # Very large models (100B+)
        
        # Quantization multipliers (relative speedup)
        quant_multipliers = {
            "Q2": 2.0,      # Fastest but lowest quality
            "Q3": 1.8,
            "Q4": 1.5,      # Good balance of speed and quality
            "Q5": 1.3,
            "Q7": 1.1,
            "Q8": 1.0,      # Reference point
            "GPTQ": 1.5,    # Similar to Q4
            "AWQ": 1.6,     # Slightly better than GPTQ
            "FP16": 0.7,    # Slower than int8 but higher quality
            "FP32": 0.4     # Slowest but highest quality
        }
        
        # CUDA vs non-CUDA (massive difference in performance)
        cuda_multiplier = 20.0 if has_cuda else 1.0
        
        # GPU tier multipliers
        gpu_tier_multipliers = {
            "consumer": 1.0,        # Reference point (e.g., RTX 3070)
            "high-end": 1.8,        # High-end consumer (e.g., RTX 4090)
            "professional": 2.5,    # Professional GPU (e.g., A100)
            "multi-gpu": 3.5        # Multiple high-end GPUs
        }
        
        # Context length adjustment (longer contexts are slower due to attention overhead)
        # This is a simplified model that approximates the effect of KV cache size on performance
        if context_length <= 1024:
            context_multiplier = 1.2  # Short contexts are faster
        elif context_length <= 4096:
            context_multiplier = 1.0  # Reference point
        elif context_length <= 8192:
            context_multiplier = 0.85  # Noticeable slowdown
        elif context_length <= 16384:
            context_multiplier = 0.7   # Significant slowdown
        elif context_length <= 32768:
            context_multiplier = 0.5   # Major slowdown
        else:
            context_multiplier = 0.3   # Extreme slowdown for very long contexts
            
        # Determine GPU tier based on VRAM requirements if not specified
        if gpu_tier == "auto":
            vram_data = calculate_vram_usage(params_billions, quant_type, context_length, has_cuda)
            vram_needed = vram_data["total"]
            if vram_needed <= 8:
                gpu_tier = "consumer"
            elif vram_needed <= 24:
                gpu_tier = "high-end"
            elif vram_needed <= 48:
                gpu_tier = "professional"
            else:
                gpu_tier = "multi-gpu"
        
        # Calculate tokens per second
        tps = base_tps * quant_multipliers[quant_type] * cuda_multiplier * gpu_tier_multipliers[gpu_tier] * context_multiplier
        
        # Round to nearest whole number for cleaner display
        return round(tps)

# Calculate requirements
if operation_mode == "Inference":
    vram_data = calculate_vram_usage(
        num_params, 
        quantization, 
        context_length, 
        cuda_support, 
        operation_mode
    )
else:  # Training mode
    vram_data = calculate_vram_usage(
        num_params, 
        quantization, 
        context_length, 
        cuda_support, 
        operation_mode,
        batch_size,
        gradient_accumulation,
        precision_type,
        optimizer_choice
    )

vram_needed = vram_data["total"]
model_weight_memory = vram_data["model_weights"]
kv_cache_memory = vram_data["kv_cache"]
overhead_memory = vram_data["overhead"]

# Training-specific memory components
optimizer_memory = vram_data.get("optimizer", 0)
gradients_memory = vram_data.get("gradients", 0)
activation_memory = vram_data.get("activations", 0)

disk_size = calculate_disk_size(num_params, quantization)
system_ram = calculate_system_ram(vram_needed)
gpu_config = recommend_gpu_config(vram_needed)
num_gpus = calculate_gpus_required(vram_needed)

# Determine GPU tier based on recommended config
if "consumer" in gpu_config.lower():
    if "high-end" in gpu_config.lower():
        gpu_tier = "high-end"
    else:
        gpu_tier = "consumer"
elif "professional" in gpu_config.lower():
    gpu_tier = "professional"
else:
    gpu_tier = "multi-gpu"

# Calculate tokens per second
tokens_per_second = estimate_tokens_per_second(num_params, quantization, context_length, cuda_support, gpu_tier)

# Display calculated requirements
st.header("Hardware Requirements")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Specifications")
    st.markdown(f"**Model Size:** {num_params} billion parameters")
    st.markdown(f"**Quantization:** {quantization}")
    st.markdown(f"**Context Length:** {context_length} tokens")
    st.markdown(f"**CUDA Support:** {'Enabled' if cuda_support else 'Disabled'}")
    st.markdown(f"**Operation Mode:** {operation_mode}")
    
    if operation_mode == "Training":
        st.markdown(f"**Batch Size:** {batch_size}")
        st.markdown(f"**Gradient Accumulation:** {gradient_accumulation}")
        st.markdown(f"**Training Precision:** {precision_type}")
        st.markdown(f"**Optimizer:** {optimizer_choice}")
    
    st.subheader("Computed Requirements")
    st.markdown(f"**Total VRAM Needed:** {vram_needed:.2f} GB")
    st.markdown(f"**— Model Weights:** {model_weight_memory:.2f} GB")
    st.markdown(f"**— KV Cache:** {kv_cache_memory:.2f} GB")
    
    if operation_mode == "Training":
        st.markdown(f"**— Optimizer States:** {optimizer_memory:.2f} GB")
        st.markdown(f"**— Gradients:** {gradients_memory:.2f} GB")
        st.markdown(f"**— Activations:** {activation_memory:.2f} GB")
        
    st.markdown(f"**— Runtime Overhead:** {overhead_memory:.2f} GB")
    st.markdown(f"**On-Disk Size:** {disk_size:.2f} GB")
    st.markdown(f"**GPU Config:** {gpu_config}")
    st.markdown(f"**Number of GPUs Required:** {num_gpus}")
    st.markdown(f"**System RAM:** {system_ram:.2f} GB")
    
    if operation_mode == "Inference":
        st.subheader("Inference Performance")
        st.markdown(f"**Estimated Tokens Per Second:** {tokens_per_second}")
    else:
        st.subheader("Training Performance")
        # Calculate approximate training time per epoch
        # This is a very rough estimate based on model size and hardware
        if cuda_support:
            if vram_needed <= 24:  # Can fit on a single high-end GPU
                tokens_per_second_training = max(1, tokens_per_second / 15)  # Training is ~15x slower than inference
            elif vram_needed <= 80:  # Single A100
                tokens_per_second_training = max(1, tokens_per_second / 12)
            else:  # Multi-GPU
                tokens_per_second_training = max(1, tokens_per_second / 10)
        else:
            tokens_per_second_training = max(0.1, tokens_per_second / 50)  # CPU training is extremely slow
            
        st.markdown(f"**Estimated Tokens Per Second (Training):** {tokens_per_second_training:.1f}")

with col2:
    # Create a unified memory comparison
    st.subheader("Unified Memory Comparison")
    
    # Get information about unified memory options
    unified_memory_options = {
        # Apple Silicon
        "Apple M1 Pro": 16,
        "Apple M1 Max": 32,
        "Apple M1 Ultra": 64,
        "Apple M2 Pro": 16,
        "Apple M2 Max": 32,
        "Apple M2 Ultra": 64,
        "Apple M3 Pro": 18,
        "Apple M3 Max": 36,
        "Apple M3 Ultra": 72,
        
        # NVIDIA GPUs (Consumer)
        "NVIDIA RTX 3060 (12GB)": 12,
        "NVIDIA RTX 3070 (8GB)": 8,
        "NVIDIA RTX 3080 (10GB)": 10,
        "NVIDIA RTX 3090 (24GB)": 24,
        "NVIDIA RTX 4060 (8GB)": 8,
        "NVIDIA RTX 4070 (12GB)": 12,
        "NVIDIA RTX 4080 (16GB)": 16,
        "NVIDIA RTX 4090 (24GB)": 24,
        
        # NVIDIA Professional
        "NVIDIA A10 (24GB)": 24,
        "NVIDIA A100 (40GB)": 40,
        "NVIDIA A100 (80GB)": 80,
        "NVIDIA H100 (80GB)": 80,
        "NVIDIA L40 (48GB)": 48,
        
        # Multi-GPU Configurations
        "2x NVIDIA A100 (80GB)": 160,
        "4x NVIDIA A100 (80GB)": 320,
        "8x NVIDIA A100 (80GB)": 640,
        "2x NVIDIA H100 (80GB)": 160,
        "4x NVIDIA H100 (80GB)": 320,
        "8x NVIDIA H100 (80GB)": 640,
        
        # Integrated Options
        "NVIDIA Grace Hopper (480GB)": 480,
        "Cloud Instance (High Memory)": 128
    }
    
    # Create a comparison dataframe
    comparison_data = []
    for device, memory in unified_memory_options.items():
        can_run = "Yes" if memory >= vram_needed else "No"
        headroom = memory - vram_needed
        headroom_pct = (headroom / memory) * 100 if memory >= vram_needed else 0
        
        comparison_data.append({
            "Device": device,
            "Unified Memory (GB)": memory,
            "Can Run Model": can_run,
            "Memory Headroom (GB)": max(0, headroom),
            "Headroom %": max(0, headroom_pct)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display the comparison table
    st.dataframe(comparison_df)
    
    # Create a visualization of the memory requirements
    # Prepare data for memory components chart
    memory_components = []
    
    # Add base components
    memory_components.append({"Component": "Model Weights", "Size (GB)": model_weight_memory, "Category": "VRAM"})
    memory_components.append({"Component": "KV Cache", "Size (GB)": kv_cache_memory, "Category": "VRAM"})
    
    # Add training-specific components if in training mode
    if operation_mode == "Training":
        memory_components.append({"Component": "Optimizer States", "Size (GB)": optimizer_memory, "Category": "VRAM"})
        memory_components.append({"Component": "Gradients", "Size (GB)": gradients_memory, "Category": "VRAM"})
        memory_components.append({"Component": "Activations", "Size (GB)": activation_memory, "Category": "VRAM"})
    
    # Add runtime overhead
    memory_components.append({"Component": "Runtime Overhead", "Size (GB)": overhead_memory, "Category": "VRAM"})
    
    # Add disk and system RAM
    memory_components.append({"Component": "Disk Space", "Size (GB)": disk_size, "Category": "Storage"})
    memory_components.append({"Component": "System RAM", "Size (GB)": system_ram, "Category": "System"})
    
    memory_df = pd.DataFrame(memory_components)
    
    # Create a figure for total requirements comparison
    fig = px.bar(
        memory_df, 
        x="Component", 
        y="Size (GB)",
        color="Category",
        title="Memory Requirements Comparison"
    )
    
    # Create a separate figure just for VRAM components
    vram_df = memory_df[memory_df["Category"] == "VRAM"].copy()
    
    # Create a pie chart for VRAM breakdown
    vram_fig = px.pie(
        vram_df,
        values="Size (GB)",
        names="Component",
        title=f"VRAM Usage Breakdown: {vram_needed:.2f} GB Total",
        hole=0.4
    )
    
    # Display charts
    st.subheader("Memory Breakdown")
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(vram_fig, use_container_width=True)
    
    with col2:
        st.plotly_chart(fig, use_container_width=True)

# Advanced Analysis Tab
with st.expander("Advanced Analysis"):
    st.subheader("Parameter vs Memory Scaling")
    
    # Create scaling analysis for different quantizations
    param_range = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
    
    scaling_data = []
    for param in param_range:
        for quant in ["Q4", "Q8", "FP16", "FP32"]:
            vram_with_cuda_data = calculate_vram_usage(param, quant, context_length, True)
            vram_without_cuda_data = calculate_vram_usage(param, quant, context_length, False)
            
            vram_with_cuda = vram_with_cuda_data["total"]
            vram_without_cuda = vram_without_cuda_data["total"]
            
            scaling_data.append({
                "Parameters (B)": param,
                "VRAM (GB)": vram_with_cuda,
                "Quantization": quant,
                "CUDA": "Enabled"
            })
            
            scaling_data.append({
                "Parameters (B)": param,
                "VRAM (GB)": vram_without_cuda,
                "Quantization": quant,
                "CUDA": "Disabled"
            })
    
    scaling_df = pd.DataFrame(scaling_data)
    
    # Add option to show/hide CUDA impact in the chart
    show_cuda_impact = st.checkbox("Show CUDA Impact on Scaling", value=False)
    
    if show_cuda_impact:
        fig = px.line(
            scaling_df, 
            x="Parameters (B)", 
            y="VRAM (GB)", 
            color="Quantization",
            line_dash="CUDA",  # Use line style to differentiate CUDA/non-CUDA
            log_x=True,
            log_y=True,
            title="Parameter vs VRAM Scaling with CUDA Impact (Log-Log)"
        )
    else:
        # Filter to just show CUDA-enabled data (default view)
        cuda_enabled_df = scaling_df[scaling_df["CUDA"] == "Enabled"]
        fig = px.line(
            cuda_enabled_df, 
            x="Parameters (B)", 
            y="VRAM (GB)", 
            color="Quantization",
            log_x=True,
            log_y=True,
            title="Parameter vs VRAM Scaling (Log-Log)"
        )
    st.plotly_chart(fig)
    
    # Comparison with known models
    st.subheader("Comparison with Known Models")
    
    known_models = {
        "GPT-3 (175B)": {"params": 175, "quant": "FP16"},
        "LLaMA 2 (7B)": {"params": 7, "quant": "Q4"},
        "LLaMA 2 (13B)": {"params": 13, "quant": "Q4"},
        "LLaMA 2 (70B)": {"params": 70, "quant": "Q4"},
        "Falcon (180B)": {"params": 180, "quant": "FP16"},
        "Gemma (7B)": {"params": 7, "quant": "Q4"},
        "Mistral (7B)": {"params": 7, "quant": "Q4"},
        "Claude 3 Opus (~300B est.)": {"params": 300, "quant": "FP16"},
        "Mixtral 8x7B": {"params": 47, "quant": "Q4"},  # MoE architecture, effective params
        "Llama 3 (8B)": {"params": 8, "quant": "Q4"},
        "Llama 3 (70B)": {"params": 70, "quant": "Q4"},
    }
    
    model_data = []
    for model, specs in known_models.items():
        vram_with_cuda_data = calculate_vram_usage(specs["params"], specs["quant"], context_length, True)
        vram_without_cuda_data = calculate_vram_usage(specs["params"], specs["quant"], context_length, False)
        
        vram_with_cuda = vram_with_cuda_data["total"]
        vram_without_cuda = vram_without_cuda_data["total"]
        
        # Calculate tokens per second for each model
        tokens_with_cuda_consumer = estimate_tokens_per_second(specs["params"], specs["quant"], context_length, True, "consumer")
        tokens_with_cuda_highend = estimate_tokens_per_second(specs["params"], specs["quant"], context_length, True, "high-end")
        tokens_with_cuda_pro = estimate_tokens_per_second(specs["params"], specs["quant"], context_length, True, "professional")
        tokens_without_cuda = estimate_tokens_per_second(specs["params"], specs["quant"], context_length, False, "consumer")
        
        model_data.append({
            "Model": model,
            "Parameters (B)": specs["params"],
            "Quantization": specs["quant"],
            "VRAM with CUDA (GB)": vram_with_cuda,
            "VRAM without CUDA (GB)": vram_without_cuda,
            "CUDA Impact (GB)": vram_with_cuda - vram_without_cuda,
            "TPS (Consumer GPU)": tokens_with_cuda_consumer,
            "TPS (High-End GPU)": tokens_with_cuda_highend,
            "TPS (Pro GPU)": tokens_with_cuda_pro,
            "TPS (No CUDA)": tokens_without_cuda,
            "Number of GPUs (A100)": max(1, int(np.ceil(vram_with_cuda / 80)))
        })
    
    model_df = pd.DataFrame(model_data)
    st.dataframe(model_df)
    
    # Add your current model to the comparison
    current_model_with_cuda = calculate_vram_usage(num_params, quantization, context_length, True)
    current_model_without_cuda = calculate_vram_usage(num_params, quantization, context_length, False)
    
    current_model = {
        "Model": f"Your Model ({num_params}B)",
        "Parameters (B)": num_params,
        "Quantization": quantization,
        "VRAM with CUDA (GB)": current_model_with_cuda["total"],
        "VRAM without CUDA (GB)": current_model_without_cuda["total"],
        "CUDA Impact (GB)": current_model_with_cuda["total"] - current_model_without_cuda["total"],
        "Number of GPUs (A100)": num_gpus
    }
    
    st.markdown("### Your Model Compared to Known Models")
    comparison_fig = go.Figure()
    
    # Add bars for known models
    # Add option to show with or without CUDA
    vram_column = "VRAM with CUDA (GB)" if cuda_support else "VRAM without CUDA (GB)"
    
    comparison_fig.add_trace(go.Bar(
        x=[m["Model"] for m in model_data],
        y=[m[vram_column] for m in model_data],
        name="Known Models",
        marker_color="lightblue"
    ))
    
    # Add bar for current model
    comparison_fig.add_trace(go.Bar(
        x=[current_model["Model"]],
        y=[current_model[vram_column]],
        name="Your Model",
        marker_color="red"
    ))
    
    comparison_fig.update_layout(
        title="VRAM Requirements Comparison",
        xaxis_title="Model",
        yaxis_title="VRAM (GB)",
        xaxis={'categoryorder':'total descending'}
    )
    
    st.plotly_chart(comparison_fig)

# Add reference information
st.markdown("---")
st.markdown("""
### KV Cache Explained:
- **What it is:** Storage of previously computed attention keys (K) and values (V) during inference
- **Why it matters:** Prevents recalculating attention for previous tokens when generating new tokens
- **Scaling:** Grows linearly with context length and model layers
- **Formula:** KV Cache Size ≈ 2 * num_layers * seq_length * hidden_size * bytes_per_element
- **Memory impact:** For long contexts (32K+), KV cache can require more memory than the model weights

### Notes on Quantization:
- **Q2-Q8**: Refers to bits per weight in quantized models
- **FP16/FP32**: Full precision floating point (16/32 bits per weight)
- **GPTQ/AWQ**: Advanced quantization methods (typically ~4 bits per weight)

### Context Length Impact:
- **Memory Usage:** Longer contexts require more VRAM for KV cache storage
- **Performance:** Very long contexts can significantly reduce inference speed
- **Tradeoffs:** Higher context enables processing more text but increases resource requirements

### CUDA Support Impact:
- **With CUDA**: Enables GPU acceleration but may require ~10% additional memory for CUDA-specific optimizations
- **Without CUDA**: May use slightly less memory but performance will be significantly slower
- CUDA support is essential for practical inference with larger models

### Tokens Per Second (TPS) Performance:
- **40+ TPS**: Excellent (real-time conversation)
- **20-40 TPS**: Very good performance
- **10-20 TPS**: Good performance
- **5-10 TPS**: Adequate for most use cases
- **1-5 TPS**: Noticeable lag in responses
- **<1 TPS**: Very slow, not suitable for interactive applications

### Hardware Recommendations:
- Models under 10B parameters can run on consumer GPUs with appropriate quantization
- 10-70B models typically require high-end GPUs or multiple GPUs
- 100B+ models generally require multiple professional GPUs or specialized hardware

### References:
- These calculations are approximations and actual requirements may vary
- KV cache calculations assume standard attention mechanisms
- Performance estimates assume standard inference settings
- Actual performance may vary based on specific hardware, software optimizations, and batch sizes
""")