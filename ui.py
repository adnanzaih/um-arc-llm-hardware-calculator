import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def create_sidebar_inputs():
    """Create sidebar inputs and return the configuration as a dictionary."""
    st.sidebar.header("Model Configuration")
    
    # Model size selection - offering both slider and direct input
    param_input_method = st.sidebar.radio(
        "Parameter Input Method",
        options=["Direct Input", "Slider"],
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

    # Data sensitivity option
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Sensitivity")
    sensitivity_level = st.sidebar.radio(
        "Data Sensitivity Level",
        options=["Moderate", "High"],
        index=0,
        help="Moderate: Only great-lakes partitions. High: Only armis2 and lighthouse (excludes great-lakes for security)"
    )

    # Training vs Inference mode
    st.sidebar.markdown("---")
    st.sidebar.subheader("Operation Settings")
    operation_mode = st.sidebar.radio(
        "Operation Mode",
        options=["Inference", "Training"],
        index=0,
        help="Inference requires less memory than training"
    )

    # Initialize the configuration dictionary
    config = {
        "num_params": num_params,
        "quantization": quantization,
        "context_length": context_length,
        "cuda_support": cuda_support,
        "operation_mode": operation_mode,
        "sensitivity_level": sensitivity_level.lower()  # Convert to lowercase for consistency
    }

    # Training-specific configuration
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
        
        # Add training-specific configuration
        config.update({
            "batch_size": batch_size,
            "gradient_accumulation": gradient_accumulation,
            "precision_type": precision_type,
            "optimizer_choice": optimizer_choice
        })

    return config

def display_requirements(
    model_config, vram_needed, model_weight_memory, kv_cache_memory, 
    overhead_memory, optimizer_memory, gradients_memory, activation_memory,
    disk_size, system_ram, gpu_config, num_gpus,
    tokens_per_second=None, tokens_per_second_training=None
):
    """Display calculated requirements in a compact table format."""
    # Create a tabular format using two columns
    col1, col2 = st.columns(2)
    
    # Create a dictionary for model specifications
    model_specs = {
        "Model Size": f"{model_config['num_params']} billion parameters",
        "Quantization": model_config['quantization'],
        "Context Length": f"{model_config['context_length']} tokens",
        "CUDA Support": 'Enabled' if model_config['cuda_support'] else 'Disabled',
        "Operation Mode": model_config['operation_mode'],
        "Data Sensitivity": model_config['sensitivity_level'].capitalize()
    }
    
    # Add training-specific configuration if in training mode
    if model_config["operation_mode"] == "Training":
        model_specs.update({
            "Batch Size": str(model_config['batch_size']),  # Convert to string
            "Gradient Accumulation": str(model_config['gradient_accumulation']),  # Convert to string
            "Training Precision": model_config['precision_type'],
            "Optimizer": model_config['optimizer_choice']
        })
    
    # Create a dictionary for computed requirements - removed partition and number of GPUs
    computed_reqs = {
        "Total VRAM Needed": f"{vram_needed:.2f} GB",
        "Model Weights": f"{model_weight_memory:.2f} GB",
        "KV Cache": f"{kv_cache_memory:.2f} GB",
        "Runtime Overhead": f"{overhead_memory:.2f} GB",
        "On-Disk Size": f"{disk_size:.2f} GB",
        "System RAM": f"{system_ram:.2f} GB"
    }
    
    # Add training-specific memory components if in training mode
    if model_config["operation_mode"] == "Training":
        computed_reqs.update({
            "Optimizer States": f"{optimizer_memory:.2f} GB",
            "Gradients": f"{gradients_memory:.2f} GB",
            "Activations": f"{activation_memory:.2f} GB"
        })
    
    # Create pandas DataFrames for better table display - ensuring all values are strings
    specs_df = pd.DataFrame(list(model_specs.items()), columns=["Specification", "Value"])
    reqs_df = pd.DataFrame(list(computed_reqs.items()), columns=["Requirement", "Value"])
    
    # Display tables side by side in columns
    with col1:
        st.markdown("#### Model Specifications")
        st.dataframe(specs_df, use_container_width=True, hide_index=True)
        
        # Note: Inference performance metrics moved to Advanced Analysis section
        # Only keep Training performance if in Training mode
        if model_config["operation_mode"] == "Training" and tokens_per_second_training is not None:
            perf_data = {"Metric": ["Estimated Tokens Per Second (Training)"], "Value": [f"{tokens_per_second_training:.1f}"]}
            perf_df = pd.DataFrame(perf_data)
            st.markdown("#### Training Performance")
            st.dataframe(perf_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### Hardware Requirements")
        st.dataframe(reqs_df, use_container_width=True, hide_index=True)

def display_memory_charts(
    operation_mode, model_weight_memory, kv_cache_memory, overhead_memory,
    optimizer_memory, gradients_memory, activation_memory,
    disk_size, system_ram, vram_needed
):
    """Create and display memory breakdown charts."""
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

def display_hardware_comparison(model_config, vram_needed, filtered_options):
    """Display hardware comparison table with focus on recommended hardware configurations."""
    st.subheader("ARC HPC Clusters Hardware Configuration")
    
    # Get number of GPUs required for each hardware option
    comparison_data = []
    for device, memory in filtered_options.items():
        # Calculate how many GPUs of this type would be needed
        gpus_needed = max(1, int(np.ceil(vram_needed / memory)))
        
        # Cap at reasonable numbers (5 is standard partition limit mentioned in calc.py)
        gpus_needed = min(gpus_needed, 5)
        
        # Calculate total memory available with this configuration
        total_memory = memory * gpus_needed
        
        # Determine if this config can run the model
        can_run = "Yes" if total_memory >= vram_needed else "No"
        
        # Calculate memory efficiency (how much of the total memory would be used)
        memory_efficiency = (vram_needed / total_memory) * 100 if total_memory > 0 else 0
        
        # Extract cluster and GPU information
        if "[" in device:
            parts = device.split("[")
            cluster = parts[0].strip()
            remaining = "[".join(parts[1:])
            gpu_type = remaining.split("]")[0].strip()
            if len(parts) > 2:
                gpu_count = parts[2].strip("[] ")
            else:
                gpu_count = "N/A"
        else:
            cluster = device
            gpu_type = "Unknown"
            gpu_count = "N/A"
        
        comparison_data.append({
            "Partition": device,
            "GPU Memory (GB)": memory,
            "GPUs Required": gpus_needed,
            "Total Memory (GB)": total_memory,
            "Can Run Model": can_run,
            "Memory Utilization": f"{memory_efficiency:.1f}%",
            "Cluster": cluster,
            "GPU Type": gpu_type,
            "Available GPUs": gpu_count
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by efficiency (most efficient first)
    comparison_df = comparison_df.sort_values(by=["Can Run Model", "GPUs Required", "Memory Utilization"], 
                                             ascending=[False, True, False])
    
    # Create tabs for compatible hardware vs all hardware
    tab1, tab2 = st.tabs(["Compatible Hardware Only", "All Hardware"])
    
    # First tab shows compatible hardware
    with tab1:
        compatible_df = comparison_df[comparison_df["Can Run Model"] == "Yes"]
        if not compatible_df.empty:
            # Select and reorder columns for better display
            display_columns = ["Partition", "GPU Memory (GB)", "GPUs Required", "Total Memory (GB)", "Memory Utilization"]
            st.dataframe(compatible_df[display_columns], use_container_width=True)
            
            # Highlight recommended configuration
            best_config = compatible_df.iloc[0]
            st.success(f"**Recommended Configuration:** {best_config['Partition']} with {best_config['GPUs Required']} GPUs, providing a total of {best_config['Total Memory (GB)']:.1f} GB GPU memory.")
        else:
            st.warning("No compatible hardware found for current configuration at this sensitivity level.")
    
    # Second tab shows all hardware
    with tab2:
        if comparison_df.empty:
            st.warning(f"No hardware options available for {model_config['sensitivity_level']} sensitivity level. Consider enabling high sensitivity.")
        else:
            # Select and reorder columns for better display
            display_columns = ["Partition", "GPU Memory (GB)", "GPUs Required", "Total Memory (GB)", "Can Run Model", "Memory Utilization"]
            st.dataframe(comparison_df[display_columns], use_container_width=True)

def display_model_comparison(model_config, known_models, vram_calc_fn, tps_calc_fn, filter_fn):
    """Create and display model comparison with sensitivity awareness."""
    with st.expander("Advanced Analysis"):
        # Add Inference Performance as the first item in the Advanced Analysis
        st.subheader("Inference Performance Analysis")
        
        # Get information about various GPU tiers
        gpu_tiers = {
            "Consumer GPU": "Standard consumer GPUs like RTX 3060, RTX 3070",
            "High-End Consumer": "High-end consumer GPUs like RTX 4080, RTX 4090",
            "Professional GPU": "Data center GPUs like A40, A100, H100",
            "No CUDA (CPU)": "CPU-only inference (no CUDA acceleration)"
        }
        
        # Calculate performance for current model on different hardware tiers
        current_model_perf = {}
        if model_config["cuda_support"]:
            current_model_perf["Consumer GPU"] = tps_calc_fn(
                model_config["num_params"], 
                model_config["quantization"], 
                model_config["context_length"], 
                True, 
                "consumer"
            )
            current_model_perf["High-End Consumer"] = tps_calc_fn(
                model_config["num_params"], 
                model_config["quantization"], 
                model_config["context_length"], 
                True, 
                "high-end"
            )
            current_model_perf["Professional GPU"] = tps_calc_fn(
                model_config["num_params"], 
                model_config["quantization"], 
                model_config["context_length"], 
                True, 
                "professional"
            )
        
        current_model_perf["No CUDA (CPU)"] = tps_calc_fn(
            model_config["num_params"], 
            model_config["quantization"], 
            model_config["context_length"], 
            False, 
            "consumer"
        )
        
        # Create performance data for visualization
        perf_data = []
        for tier, tps in current_model_perf.items():
            perf_data.append({
                "Hardware Tier": tier,
                "Tokens Per Second": tps,
                "Description": gpu_tiers.get(tier, "")
            })
        
        perf_df = pd.DataFrame(perf_data)
        
        # Display the performance information in a table
        st.markdown(f"#### Performance of Your {model_config['num_params']}B Parameter Model")
        st.dataframe(perf_df[["Hardware Tier", "Tokens Per Second", "Description"]], use_container_width=True, hide_index=True)
        
        # Create a performance visualization
        fig = px.bar(
            perf_df,
            x="Hardware Tier",
            y="Tokens Per Second",
            title=f"Estimated Performance with {model_config['quantization']} Quantization",
            color="Tokens Per Second",
            color_continuous_scale="Viridis",
            text="Tokens Per Second"
        )
        
        fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add performance interpretation
        st.markdown("#### Performance Interpretation")
        st.markdown("""
        - **40+ TPS**: Excellent (real-time conversation)
        - **20-40 TPS**: Very good performance
        - **10-20 TPS**: Good performance
        - **5-10 TPS**: Adequate for most use cases
        - **1-5 TPS**: Noticeable lag in responses
        - **<1 TPS**: Very slow, not suitable for interactive applications
        """)
        
        # Now continue with the rest of the advanced analysis content
        st.markdown("---")
        st.subheader("Parameter vs Memory Scaling")
        
        # Create scaling analysis for different quantizations
        param_range = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
        
        scaling_data = []
        for param in param_range:
            for quant in ["Q4", "Q8", "FP16", "FP32"]:
                vram_with_cuda_data = vram_calc_fn(param, quant, model_config["context_length"], True)
                vram_without_cuda_data = vram_calc_fn(param, quant, model_config["context_length"], False)
                
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
        
        # Get the filtered hardware options for GPU calculations
        filtered_hardware = filter_fn(model_config["sensitivity_level"])
        max_vram_per_gpu = max(filtered_hardware.values()) if filtered_hardware else 80
        
        model_data = []
        for model, specs in known_models.items():
            vram_with_cuda_data = vram_calc_fn(specs["params"], specs["quant"], model_config["context_length"], True)
            vram_without_cuda_data = vram_calc_fn(specs["params"], specs["quant"], model_config["context_length"], False)
            
            vram_with_cuda = vram_with_cuda_data["total"]
            vram_without_cuda = vram_without_cuda_data["total"]
            
            # Calculate tokens per second for each model
            tokens_with_cuda_consumer = tps_calc_fn(specs["params"], specs["quant"], model_config["context_length"], True, "consumer")
            tokens_with_cuda_highend = tps_calc_fn(specs["params"], specs["quant"], model_config["context_length"], True, "high-end")
            tokens_with_cuda_pro = tps_calc_fn(specs["params"], specs["quant"], model_config["context_length"], True, "professional")
            tokens_without_cuda = tps_calc_fn(specs["params"], specs["quant"], model_config["context_length"], False, "consumer")
            
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
                "Number of GPUs Required": max(1, int(np.ceil(vram_with_cuda / max_vram_per_gpu)))
            })
        
        model_df = pd.DataFrame(model_data)
        
        # Add your current model to the comparison
        current_model_with_cuda = vram_calc_fn(model_config["num_params"], model_config["quantization"], model_config["context_length"], True)
        current_model_without_cuda = vram_calc_fn(model_config["num_params"], model_config["quantization"], model_config["context_length"], False)
        
        # Create performance comparison chart
        if model_config["cuda_support"]:
            perf_column = "TPS (High-End GPU)"  # Default to high-end GPU with CUDA
        else:
            perf_column = "TPS (No CUDA)"
        
        # Add current model to the performance dataframe
        current_perf = {
            "Model": f"Your Model ({model_config['num_params']}B)",
            "Parameters (B)": model_config["num_params"],
            "Quantization": model_config["quantization"],
            "VRAM with CUDA (GB)": current_model_with_cuda["total"],
            "VRAM without CUDA (GB)": current_model_without_cuda["total"],
            "Number of GPUs Required": max(1, int(np.ceil(current_model_with_cuda["total"] / max_vram_per_gpu)))
        }
        
        if model_config["cuda_support"]:
            current_perf["TPS (Consumer GPU)"] = tps_calc_fn(model_config["num_params"], model_config["quantization"], model_config["context_length"], True, "consumer")
            current_perf["TPS (High-End GPU)"] = tps_calc_fn(model_config["num_params"], model_config["quantization"], model_config["context_length"], True, "high-end")
            current_perf["TPS (Pro GPU)"] = tps_calc_fn(model_config["num_params"], model_config["quantization"], model_config["context_length"], True, "professional")
        else:
            current_perf["TPS (No CUDA)"] = tps_calc_fn(model_config["num_params"], model_config["quantization"], model_config["context_length"], False, "consumer")
        
        # Show sensitivity information
        st.markdown(f"### Model Comparison (For {model_config['sensitivity_level'].capitalize()} Sensitivity Level)")
        st.markdown(f"GPU calculations are based on the maximum VRAM capacity of {max_vram_per_gpu}GB available at this sensitivity level.")
        
        # Show performance comparison table
        if model_config["cuda_support"]:
            # Create a table comparing tokens per second across different GPU tiers
            performance_columns = ["Model", "Parameters (B)", "TPS (Consumer GPU)", 
                                "TPS (High-End GPU)", "TPS (Pro GPU)", "Number of GPUs Required"]
            perf_df = pd.DataFrame([{k: v for k, v in m.items() if k in performance_columns} 
                                  for m in model_data])
        else:
            # When CUDA is disabled, just show the no-CUDA performance
            performance_columns = ["Model", "Parameters (B)", "TPS (No CUDA)", "Number of GPUs Required"]
            perf_df = pd.DataFrame([{k: v for k, v in m.items() if k in performance_columns} 
                                  for m in model_data])
        
        # Extract performance data for the current model
        current_model_perf = {k: v for k, v in current_perf.items() if k in performance_columns}
        
        # Add current model to performance dataframe
        perf_df = pd.concat([perf_df, pd.DataFrame([current_model_perf])], ignore_index=True)
        
        # Sort by parameter count for easier comparison
        perf_df = perf_df.sort_values("Parameters (B)")
        
        # Display the performance table
        st.dataframe(perf_df)
        
        # Create TPS comparison chart
        if model_config["cuda_support"]:
            perf_column = "TPS (High-End GPU)"
        else:
            perf_column = "TPS (No CUDA)"
            
        if perf_column in perf_df.columns:
            tps_fig = px.bar(
                perf_df,
                x="Model",
                y=perf_column,
                color="Parameters (B)",
                color_continuous_scale="Viridis",
                title=f"Tokens Per Second Comparison ({perf_column})"
            )
            
            tps_fig.update_layout(
                xaxis={'categoryorder':'total descending'},
                yaxis_title="Tokens Per Second"
            )
            
            st.plotly_chart(tps_fig)
        
        st.markdown("### VRAM Requirements Comparison")
        # Add option to show with or without CUDA
        vram_column = "VRAM with CUDA (GB)" if model_config["cuda_support"] else "VRAM without CUDA (GB)"
            
        # Create the comparison figure
        comparison_fig = go.Figure()
        
        # Add bars for known models
        comparison_fig.add_trace(go.Bar(
            x=[m["Model"] for m in model_data],
            y=[m[vram_column] for m in model_data],
            name="Known Models",
            marker_color="lightblue"
        ))
        
        # Add bar for current model
        comparison_fig.add_trace(go.Bar(
            x=[f"Your Model ({model_config['num_params']}B)"],
            y=[current_model_with_cuda["total"] if model_config["cuda_support"] else current_model_without_cuda["total"]],
            name="Your Model",
            marker_color="red"
        ))
        
        comparison_fig.update_layout(
            title=f"VRAM Requirements Comparison ({model_config['sensitivity_level'].capitalize()} Sensitivity)",
            xaxis_title="Model",
            yaxis_title="VRAM (GB)",
            xaxis={'categoryorder':'total descending'}
        )
        
        st.plotly_chart(comparison_fig)