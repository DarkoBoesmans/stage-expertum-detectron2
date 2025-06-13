#!/usr/bin/env python3
"""
Unified Training Script for Detectron2 Object Detection Models
----------------------------------------------------------
This script combines the functionality of train.py and check_model.py,
providing a single interface for both regular training and memory-efficient
training with monitoring. It automatically adapts to the available hardware
and user preferences.

The script supports training with minimal memory usage by breaking down
training into mini-batches of iterations with memory cleanup in between.
"""

import argparse  # For parsing command-line arguments
import os  # For file and directory operations
import util  # Custom utility module for training functions
import torch  # PyTorch deep learning framework
import gc  # Garbage collection for memory management
import sys  # System-specific parameters and functions
import numpy as np  # Numerical operations library
import psutil  # For system and process monitoring
import time  # For time-related functions
import datetime  # For timestamped logging
from threading import Thread  # For parallel execution
import platform  # For system information

# Global flag to signal when memory monitoring should stop
exit_monitoring = False


def log_with_timestamp(message):
    """
    Print a message with a timestamp prefix for better log readability.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def monitor_memory_usage():
    """
    Monitor and print memory usage of the current process and system.
    Runs in a separate thread and updates every 10 seconds.
    """
    log_with_timestamp("Memory monitoring started")
    while not exit_monitoring:
        # Get current process information
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        # Convert memory usage to gigabytes for readability
        memory_gb = memory_info.rss / (1024 * 1024 * 1024)

        # Get overall system memory information
        system_memory = psutil.virtual_memory()
        system_memory_used_percent = system_memory.percent
        system_memory_available_gb = system_memory.available / (1024 * 1024 * 1024)

        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)

        # Print current memory and CPU usage statistics
        log_with_timestamp(
            f"Resource Monitor | Process: {memory_gb:.2f}GB | System used: {system_memory_used_percent}% | "
            f"Available: {system_memory_available_gb:.2f}GB | CPU: {cpu_percent}%"
        )
        time.sleep(10)  # Wait 10 seconds before next update

    log_with_timestamp("Memory monitoring stopped")


def detect_hardware():
    """
    Detect hardware capabilities to suggest optimal training parameters.
    Returns a dictionary with hardware information.
    """
    hw_info = {
        "cpu": {
            "cores": os.cpu_count(),
            "model": platform.processor(),
        },
        "memory": {
            "total_gb": psutil.virtual_memory().total / (1024 * 1024 * 1024),
            "available_gb": psutil.virtual_memory().available / (1024 * 1024 * 1024),
        },
        "gpu": {
            "available": torch.cuda.is_available(),
            "count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        },
    }
    
    # Determine recommended settings based on hardware
    if hw_info["gpu"]["available"]:
        hw_info["recommended_device"] = "cuda"
        hw_info["recommended_batch_size"] = 2  # Conservative default for GPU
        hw_info["recommended_mini_batch_size"] = 10
    else:
        hw_info["recommended_device"] = "cpu"
        
        # For CPUs, set conservative defaults based on available memory
        if hw_info["memory"]["total_gb"] < 8:  # Low memory system
            hw_info["recommended_batch_size"] = 1
            hw_info["recommended_mini_batch_size"] = 2
        elif hw_info["memory"]["total_gb"] < 16:  # Mid-range system
            hw_info["recommended_batch_size"] = 1
            hw_info["recommended_mini_batch_size"] = 5
        else:  # High-end system
            hw_info["recommended_batch_size"] = 2
            hw_info["recommended_mini_batch_size"] = 10
    
    return hw_info


def train_unified(
    output_dir,  # Directory to save model checkpoints
    data_dir,  # Directory containing training data
    class_list,  # File containing class names
    device="cpu",  # Device to use for training (cpu or cuda)
    learning_rate=0.00005,  # Learning rate for optimizer
    batch_size=1,  # Number of samples per batch
    iterations=50,  # Total number of training iterations
    checkpoint_period=25,  # Save model every this many iterations
    model="COCO-Detection/retinanet_R_50_FPN_1x.yaml",  # Model configuration
    mini_batch_size=5,  # Number of iterations to run before freeing memory
    monitor_resources=True,  # Whether to monitor memory and CPU usage
    memory_efficient=True,  # Whether to use memory-efficient training
):
    """
    Unified training function that can perform both standard training and
    memory-efficient training with monitoring.
    
    This function can either:
    1. Run full training in one go (standard Detectron2 approach)
    2. Break training into mini-batches with memory cleanup (memory-efficient)
    
    Args:
        output_dir (str): Directory where model checkpoints and final model will be saved
        data_dir (str): Directory containing the training data organized in train/imgs and train/anns
        class_list (str): Path to a file containing the class names, one per line
        device (str): Device to use for training, either "cpu" or "cuda"
        learning_rate (float): Learning rate for the optimizer during training
        batch_size (int): Number of images processed in a single training batch
        iterations (int): Total number of training iterations to perform
        checkpoint_period (int): Save a model checkpoint every this many iterations
        model (str): Name of the model configuration from Detectron2 model zoo
        mini_batch_size (int): Number of iterations to run before memory cleanup
        monitor_resources (bool): Whether to monitor memory and CPU usage
        memory_efficient (bool): Whether to use memory-efficient training
    """
    # Clean up memory before starting training
    gc.collect()
    # Clear GPU memory if available
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Start memory monitoring in a separate thread if requested
    global exit_monitoring
    exit_monitoring = False
    monitor_thread = None
    
    if monitor_resources:
        monitor_thread = Thread(target=monitor_memory_usage)
        monitor_thread.daemon = True  # Thread will exit when main program exits
        monitor_thread.start()

    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Print training configuration information
        log_with_timestamp(f"Starting training with the following configuration:")
        log_with_timestamp(f"  - Device: {device}")
        log_with_timestamp(f"  - Batch size: {batch_size}")
        log_with_timestamp(f"  - Learning rate: {learning_rate}")
        log_with_timestamp(f"  - Iterations: {iterations}")
        log_with_timestamp(f"  - Model: {model}")
        log_with_timestamp(f"  - Memory-efficient mode: {memory_efficient}")
        
        if memory_efficient:
            log_with_timestamp(f"  - Mini-batch size: {mini_batch_size} iterations")
            
            # Initialize counters for the training loop
            remaining_iters = iterations
            current_iter = 0

            # Continue training until all iterations are completed
            while remaining_iters > 0:
                # Calculate how many iterations to run in this mini-batch
                iters_this_round = min(mini_batch_size, remaining_iters)
                log_with_timestamp(
                    f"Training mini-batch {current_iter} to {current_iter + iters_this_round} of {iterations}"
                )

                # Free memory before each mini-batch
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

                # Set different random seed for each mini-batch for better randomization
                np.random.seed(42 + current_iter)

                try:
                    # Run training for this mini-batch
                    util.train(
                        output_dir,
                        data_dir,
                        class_list,
                        device=device,
                        learning_rate=float(learning_rate),
                        batch_size=int(batch_size),
                        iterations=int(iters_this_round),
                        checkpoint_period=int(checkpoint_period),
                        model=model,
                    )

                    # Update progress counters
                    current_iter += iters_this_round
                    remaining_iters -= iters_this_round

                    log_with_timestamp(f"Completed iterations up to {current_iter}/{iterations}")

                    # Additional memory cleanup after mini-batch
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

                    # Pause briefly to allow system resources to stabilize
                    time.sleep(2)

                except RuntimeError as e:
                    # Handle out-of-memory errors by reducing mini-batch size
                    if "out of memory" in str(e) or "CUDA out of memory" in str(e):
                        log_with_timestamp(f"WARNING: Out of memory error: {str(e)}")
                        log_with_timestamp(f"Trying to reduce mini-batch size...")

                        if mini_batch_size > 1:
                            # Cut mini-batch size in half, but keep minimum of 1
                            mini_batch_size = max(1, mini_batch_size // 2)
                            log_with_timestamp(
                                f"Reduced mini-batch size to {mini_batch_size}, trying again..."
                            )
                            # Don't increment counters, retry with smaller mini-batch
                            gc.collect()
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None
                            time.sleep(5)  # Wait longer to ensure memory is cleared
                            continue
                        else:
                            # Can't reduce mini-batch size further
                            log_with_timestamp(
                                "ERROR: Already at minimum mini-batch size. Training cannot proceed."
                            )
                            log_with_timestamp("Try using a smaller model or increase system memory.")
                            break
                    else:
                        # For other runtime errors, propagate the exception
                        log_with_timestamp(f"ERROR: {str(e)}")
                        raise e

            log_with_timestamp(f"Training completed: {current_iter}/{iterations} iterations")
        
        else:
            # Run standard training without mini-batches
            log_with_timestamp("Running standard training without mini-batches")
            
            util.train(
                output_dir,
                data_dir,
                class_list,
                device=device,
                learning_rate=float(learning_rate),
                batch_size=int(batch_size),
                iterations=int(iterations),
                checkpoint_period=int(checkpoint_period),
                model=model,
            )
            
            log_with_timestamp(f"Training completed: {iterations} iterations")

    except Exception as e:
        # Handle any other exceptions
        log_with_timestamp(f"Error during training: {str(e)}")
        raise e
    finally:
        # Ensure memory monitoring thread stops when function exits
        if monitor_thread:
            exit_monitoring = True
            monitor_thread.join(timeout=1.0)  # Wait for thread to finish, but don't block indefinitely


def suggest_optimal_parameters(hw_info):
    """
    Suggest optimal training parameters based on the detected hardware.
    """
    suggestions = {
        "device": hw_info["recommended_device"],
        "batch_size": hw_info["recommended_batch_size"],
        "mini_batch_size": hw_info["recommended_mini_batch_size"],
        
        # Model suggestions based on hardware
        "model": "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"  # Default suggestion
    }
    
    # Suggest more complex model for powerful systems
    if hw_info["gpu"]["available"] or hw_info["memory"]["total_gb"] >= 16:
        suggestions["model"] = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    
    # For very powerful GPUs, suggest even more powerful models
    if hw_info["gpu"]["available"] and any("RTX" in gpu_name for gpu_name in hw_info["gpu"]["names"]):
        suggestions["model"] = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
        suggestions["batch_size"] = 4
    
    return suggestions


if __name__ == "__main__":
    # Detect available hardware
    hw_info = detect_hardware()
    
    # Get optimal parameter suggestions based on hardware
    suggestions = suggest_optimal_parameters(hw_info)
    
    # Create command-line argument parser
    parser = argparse.ArgumentParser(
        description="Unified training script for object detection models"
    )
    
    # Hardware information arguments
    hw_group = parser.add_argument_group('Hardware Information')
    hw_group.add_argument(
        "--show-hardware-info",
        action="store_true",
        help="Show detailed hardware information and exit",
    )
    
    # Define command-line arguments with hardware-optimized default values
    parser.add_argument(
        "--class-list",
        default="./class.names",
        help="Path to file containing class names",
    )
    parser.add_argument(
        "--data-dir", 
        default="./data", 
        help="Directory containing training data"
    )
    parser.add_argument(
        "--output-dir", 
        default="./output", 
        help="Directory to save model outputs"
    )
    parser.add_argument(
        "--device", 
        default=suggestions["device"], 
        help="Device to use for training (cpu or cuda)"
    )
    parser.add_argument(
        "--learning-rate", 
        default=0.00005, 
        help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--batch-size", 
        default=suggestions["batch_size"], 
        type=int, 
        help="Batch size for training"
    )
    parser.add_argument(
        "--iterations", 
        default=1000, 
        type=int, 
        help="Total number of training iterations"
    )
    parser.add_argument(
        "--checkpoint-period", 
        default=100, 
        type=int, 
        help="Save checkpoint every N iterations"
    )
    parser.add_argument(
        "--mini-batch-size",
        default=suggestions["mini_batch_size"],
        type=int,
        help="Number of iterations to train in each mini-batch",
    )
    parser.add_argument(
        "--model",
        default=suggestions["model"],
        help="Model configuration file",
    )
    
    # Training mode options
    mode_group = parser.add_argument_group('Training Mode Options')
    mode_group.add_argument(
        "--memory-efficient",
        action="store_true",
        default=True,
        help="Use memory-efficient training with mini-batches and cleanup",
    )
    mode_group.add_argument(
        "--standard-training",
        action="store_true",
        help="Use standard training without mini-batches (disables memory-efficient mode)",
    )
    mode_group.add_argument(
        "--monitor-resources",
        action="store_true",
        default=True,
        help="Monitor system resources during training",
    )

    # Parse arguments
    args = parser.parse_args()
    
    # Process special flags first
    if args.show_hardware_info:
        print("\n===== Hardware Information =====")
        print(f"CPU: {hw_info['cpu']['model']} ({hw_info['cpu']['cores']} cores)")
        print(f"Memory: {hw_info['memory']['total_gb']:.1f}GB total, {hw_info['memory']['available_gb']:.1f}GB available")
        
        if hw_info['gpu']['available']:
            print(f"GPU: {', '.join(hw_info['gpu']['names'])}")
        else:
            print("GPU: Not available")
            
        print(f"\nRecommended settings for this hardware:")
        print(f"  --device {suggestions['device']}")
        print(f"  --batch-size {suggestions['batch_size']}")
        print(f"  --mini-batch-size {suggestions['mini_batch_size']}")
        print(f"  --model {suggestions['model']}")
        sys.exit(0)

    # If standard training is explicitly requested, disable memory-efficient mode
    if args.standard_training:
        args.memory_efficient = False

    # Print hardware information
    print("\n===== Hardware Information =====")
    print(f"CPU: {hw_info['cpu']['model']} ({hw_info['cpu']['cores']} cores)")
    print(f"Memory: {hw_info['memory']['total_gb']:.1f}GB total, {hw_info['memory']['available_gb']:.1f}GB available")
    
    if hw_info['gpu']['available']:
        print(f"GPU: {', '.join(hw_info['gpu']['names'])}")
        if args.device == "cpu":
            print("NOTE: GPU is available but you've selected CPU for training.")
            print("      To use GPU, specify --device cuda")
    else:
        print("GPU: Not available")
        if args.device == "cuda":
            print("WARNING: You specified --device cuda but no GPU is available.")
            print("         Switching to CPU training.")
            args.device = "cpu"
    
    print("\n===== Training Configuration =====")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Class list: {args.class_list}")
    print(f"Device: {args.device}")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Iterations: {args.iterations}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Training mode: {'Memory-efficient' if args.memory_efficient else 'Standard'}")
    
    if args.memory_efficient:
        print(f"Mini-batch size: {args.mini_batch_size} iterations")
    
    print("\nStarting training...\n")
    
    # Start training with the specified arguments
    train_unified(
        args.output_dir,
        args.data_dir,
        args.class_list,
        device=args.device,
        learning_rate=float(args.learning_rate),
        batch_size=args.batch_size,
        iterations=args.iterations,
        checkpoint_period=args.checkpoint_period,
        model=args.model,
        mini_batch_size=args.mini_batch_size,
        monitor_resources=args.monitor_resources,
        memory_efficient=args.memory_efficient
    )
