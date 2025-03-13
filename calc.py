import numpy as np

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

def recommend_gpu_config(vram_needed, sensitivity_level="moderate"):
    """
    Recommend hardware partition based on VRAM requirements using available cluster options.
    Prioritizes using fewer GPUs of higher memory capacity.
    
    Args:
        vram_needed: VRAM needed in GB
        sensitivity_level: 'moderate' or 'high' - determines which clusters are available
    """
    # Get available hardware options
    hardware_options = get_hardware_options()
    
    # Filter options based on sensitivity level
    filtered_options = {}
    for option_name, vram in hardware_options.items():
        if sensitivity_level == "moderate" and "great-lakes" in option_name:
            filtered_options[option_name] = vram
        elif sensitivity_level == "high":
            filtered_options[option_name] = vram
    
    # If no options available after filtering, use a default message
    if not filtered_options:
        return "No suitable partition found for current sensitivity level - consider enabling high-sensitivity partitions"
    
    # Find the best hardware configuration
    best_configs = []
    for option_name, vram in filtered_options.items():
        # Calculate how many GPUs of this type would be needed
        gpus_needed = max(1, int(np.ceil(vram_needed / vram)))
        
        # Cap at reasonable numbers (5 is standard partition limit)
        gpus_needed = min(gpus_needed, 5)
        
        # Check if this configuration can run the model
        if vram * gpus_needed >= vram_needed:
            # Calculate memory efficiency (how much of the total memory would be used)
            memory_efficiency = (vram_needed / (vram * gpus_needed)) * 100
            
            best_configs.append({
                "option_name": option_name,
                "vram": vram,
                "gpus_needed": gpus_needed,
                "total_memory": vram * gpus_needed,
                "memory_efficiency": memory_efficiency
            })
    
    # If no viable configuration found
    if not best_configs:
        if sensitivity_level == "moderate":
            return "No suitable moderate-sensitivity partition found - consider enabling high-sensitivity partitions"
        else:
            return f"No suitable configuration found - model requires more than {vram_needed:.1f} GB memory"
    
    # Sort configurations by:
    # 1. Number of GPUs (ascending - fewer is better)
    # 2. Memory efficiency (descending - higher is better)
    best_configs.sort(key=lambda x: (x["gpus_needed"], -x["memory_efficiency"]))
    
    # Return the best match
    best_config = best_configs[0]
    
    # Special case for multi-GPU configurations
    if best_config["gpus_needed"] > 1:
        return f"{best_config['option_name']} with {best_config['gpus_needed']} GPUs"
    else:
        return best_config["option_name"]

def calculate_gpus_required(vram_needed, sensitivity_level="moderate"):
    """
    Calculate number of GPUs needed based on VRAM requirements and available hardware.
    This matches the logic in recommend_gpu_config() for consistency.
    
    Args:
        vram_needed: VRAM needed in GB
        sensitivity_level: 'moderate' or 'high' - determines which clusters are available
    """
    # Get available hardware options
    hardware_options = get_hardware_options()
    
    # Filter options based on sensitivity level
    filtered_options = {}
    for option_name, vram in hardware_options.items():
        if sensitivity_level == "moderate" and "great-lakes" in option_name:
            filtered_options[option_name] = vram
        elif sensitivity_level == "high":
            filtered_options[option_name] = vram
    
    # If no options available after filtering, use a default value
    if not filtered_options:
        return max(1, min(5, int(np.ceil(vram_needed / 16))))  # Default to 16GB GPUs
    
    # Find the best hardware configuration using same logic as recommend_gpu_config
    best_configs = []
    for option_name, vram in filtered_options.items():
        # Calculate how many GPUs of this type would be needed
        gpus_needed = max(1, int(np.ceil(vram_needed / vram)))
        
        # Cap at reasonable numbers (5 is standard partition limit)
        gpus_needed = min(gpus_needed, 5)
        
        # Check if this configuration can run the model
        if vram * gpus_needed >= vram_needed:
            # Calculate memory efficiency
            memory_efficiency = (vram_needed / (vram * gpus_needed)) * 100
            
            best_configs.append({
                "option_name": option_name,
                "vram": vram,
                "gpus_needed": gpus_needed,
                "memory_efficiency": memory_efficiency
            })
    
    # If no viable configuration found
    if not best_configs:
        return max(1, min(5, int(np.ceil(vram_needed / max(filtered_options.values())))))
    
    # Sort configurations by:
    # 1. Number of GPUs (ascending - fewer is better)
    # 2. Memory efficiency (descending - higher is better)
    best_configs.sort(key=lambda x: (x["gpus_needed"], -x["memory_efficiency"]))
    
    # Return the number of GPUs needed for the best configuration
    return best_configs[0]["gpus_needed"]

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
        
    # Calculate tokens per second
    tps = base_tps * quant_multipliers[quant_type] * cuda_multiplier * gpu_tier_multipliers[gpu_tier] * context_multiplier
    
    # Round to nearest whole number for cleaner display
    return round(tps)

def get_hardware_options():
    """Return a dictionary of hardware options and their memory sizes with partition info."""
    return {
        # Cluster GPU options from cluster_gpu_counts.csv with partition info
        "great-lakes [gpu] - V100 (16GB) [52 GPUs]": 16,
        "great-lakes [spgpu] - A40 (48GB) [240 GPUs]": 48,
        "great-lakes [spgpu2] - L40S (48GB) [192 GPUs]": 48,
        "great-lakes [viz] - P40 (24GB) [4 GPUs]": 24,
        "great-lakes [gpu_mig40] - A100 2-way MIG (40GB) [16 GPUs]": 40,
        "armis2 [gpu] - V100 (16GB) [15 GPUs]": 16,
        "armis2 [precisionhealth] - GeForce RTX 2080 Ti (11GB) [48 GPUs]": 11,
        "armis2 [miumbrella] - A30 (24GB) [3 GPUs]": 24,
        "armis2 [tocho] - L40S (48GB) [48 GPUs]": 48,
        "lighthouse [tomers] - A100 (80GB) [4 GPUs]": 80,
        "lighthouse [drjieliu-v100] - V100 (32GB) [6 GPUs]": 32,
        "lighthouse [drjieliu-h100] - H100 (80GB) [8 GPUs]": 80,
        "lighthouse [mcity_project] - L40S (48GB) [16 GPUs]": 48,
        "lighthouse [krmahesh-a100] - A100 (80GB) [20 GPUs]": 80,
        "lighthouse [project_l] - H100 (80GB) [16 GPUs]": 80,
        "lighthouse [ramanvr] - H100 (80GB) [96 GPUs]": 80,
        "lighthouse [stellayu] - A40 (48GB) [32 GPUs]": 48,
        "lighthouse [venkvis-h100] - H100 (80GB) [12 GPUs]": 80,
        "lighthouse [vvh-l40s] - L40S (48GB) [8 GPUs]": 48
    }

def filter_hardware_options(sensitivity_level="moderate"):
    """
    Filter hardware options based on sensitivity level.
    
    Args:
        sensitivity_level: 'moderate' or 'high'
        
    Returns:
        Dictionary of filtered hardware options
    """
    all_options = get_hardware_options()
    
    if sensitivity_level == "moderate":
        # Only include great-lakes partitions for moderate sensitivity
        return {k: v for k, v in all_options.items() if "great-lakes" in k}
    else:
        # For high sensitivity, exclude great-lakes partitions
        return {k: v for k, v in all_options.items() if "great-lakes" not in k}

def get_known_models():
    """Return a dictionary of known models with their parameters and quantization."""
    return {
        # Smaller models (1-3B)
        "TinyLlama (1.1B)": {"params": 1.1, "quant": "Q4"},
        "Phi-1.5 (1.3B)": {"params": 1.3, "quant": "Q4"},
        "Phi-2 (2.7B)": {"params": 2.7, "quant": "Q4"},
        "Gemma (2B)": {"params": 2, "quant": "Q4"},
        
        # Medium models (4-10B)
        "LLaMA (7B)": {"params": 7, "quant": "Q4"},
        "LLaMA 2 (7B)": {"params": 7, "quant": "Q4"},
        "Mistral (7B)": {"params": 7, "quant": "Q4"},
        "Gemma (7B)": {"params": 7, "quant": "Q4"},
        "Llama 3 (8B)": {"params": 8, "quant": "Q4"},
        
        # Larger models (11-20B)
        "LLaMA (13B)": {"params": 13, "quant": "Q4"},
        "LLaMA 2 (13B)": {"params": 13, "quant": "Q4"},
        "Mistral Large (18B)": {"params": 18, "quant": "Q4"},
        
        # Large models (30-70B)
        "LLaMA (30B)": {"params": 30, "quant": "Q4"},
        "LLaMA (65B)": {"params": 65, "quant": "Q4"},
        "LLaMA 2 (70B)": {"params": 70, "quant": "Q4"},
        "Llama 3 (70B)": {"params": 70, "quant": "Q4"},
        
        # Mixture of Experts models
        "Mixtral 8x7B": {"params": 47, "quant": "Q4"},  # MoE architecture, effective params
        
        # Commercial models for reference
        "GPT-3 (175B)": {"params": 175, "quant": "FP16"},
        "Claude 3.7 Sonnet (~140B est.)": {"params": 140, "quant": "FP16"},
        "Claude 3 Opus (~300B est.)": {"params": 300, "quant": "FP16"},
    }