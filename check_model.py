import argparse  # For parsing command-line arguments
import os  # For file and directory operations
import util  # Custom utility module for training functions
import torch  # PyTorch deep learning framework
import gc  # Garbage collection for memory management
import sys  # System-specific parameters and functions
import numpy as np  # Numerical operations library
import psutil  # For system and process monitoring
import time  # For time-related functions
from threading import Thread  # For parallel execution
import datetime  # For timestamped logging

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

        # Print current memory usage statistics
        log_with_timestamp(
            f"Memory Monitor | Process: {memory_gb:.2f}GB | System used: {system_memory_used_percent}% | Available: {system_memory_available_gb:.2f}GB"
        )
        time.sleep(10)  # Wait 10 seconds before next update

    log_with_timestamp("Memory monitoring stopped")


def train_minibatches(
    output_dir,  # Directory to save model checkpoints
    data_dir,  # Directory containing training data
    class_list,  # File containing class names
    device="cpu",  # Device to use for training (cpu or cuda)
    learning_rate=0.00005,  # Learning rate for optimizer
    batch_size=1,  # Number of samples per batch
    iterations=50,  # Total number of training iterations
    checkpoint_period=25,  # Save model every this many iterations
    model="COCO-Detection/retinanet_R_50_FPN_1x.yaml",  # Model configuration
    mini_batch_size=1,  # Number of iterations to run before freeing memory - reduced default
):
    """
    Train a detection model using very small batches to minimize memory usage.
    This is especially useful for machines with limited memory or for large models.

    This function breaks down training into mini-batches of iterations, performing
    memory cleanup between each mini-batch to prevent out-of-memory errors.

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
        mini_batch_size (int): Number of iterations to run before performing memory cleanup

    Returns:
        None. The function saves the trained model to the output directory.
    """
    # Log start of training process
    log_with_timestamp("=== BEGINNING TRAINING PROCESS ===")
    log_with_timestamp("Cleaning up memory before starting training...")

    # Clean up memory before starting training
    gc.collect()
    # Clear GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        log_with_timestamp("CUDA cache cleared")

    # Start memory monitoring in a separate thread
    global exit_monitoring
    exit_monitoring = False
    log_with_timestamp("Starting memory monitoring thread...")
    monitor_thread = Thread(target=monitor_memory_usage)
    monitor_thread.daemon = True  # Thread will exit when main program exits
    monitor_thread.start()

    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        log_with_timestamp(f"Output directory ensured: {output_dir}")

        # Print training configuration information
        log_with_timestamp("=== TRAINING CONFIGURATION ===")
        log_with_timestamp(f"Device: {device}")
        log_with_timestamp(f"Batch size: {batch_size}")
        log_with_timestamp(f"Learning rate: {learning_rate}")
        log_with_timestamp(f"Total iterations: {iterations}")
        log_with_timestamp(f"Training in mini-batches of {mini_batch_size} iterations")
        log_with_timestamp(f"Model: {model}")
        log_with_timestamp(f"Class list: {class_list}")
        log_with_timestamp(f"Data directory: {data_dir}")
        log_with_timestamp(f"Checkpoint period: Every {checkpoint_period} iterations")
        log_with_timestamp("==============================")

        # Initialize counters for the training loop
        remaining_iters = iterations
        current_iter = 0

        # Continue training until all iterations are completed
        while remaining_iters > 0:
            # Calculate how many iterations to run in this mini-batch
            iters_this_round = min(mini_batch_size, remaining_iters)
            log_with_timestamp(
                f"===== TRAINING MINI-BATCH {current_iter} to {current_iter + iters_this_round} of {iterations} ====="
            )

            # Free memory before each mini-batch
            log_with_timestamp("Freeing memory before mini-batch...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                log_with_timestamp("CUDA cache cleared")

            # Set different random seed for each mini-batch for better randomization
            np.random.seed(42 + current_iter)
            log_with_timestamp(f"Set random seed to: {42 + current_iter}")

            try:
                # Run training for this mini-batch with aggressive memory management
                log_with_timestamp(
                    f"Starting training for {iters_this_round} iterations..."
                )

                # Set a lower memory limit for dataloader workers if any
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

                start_time = time.time()
                try:
                    # Temporarily reduce PyTorch's internal cache size
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        # Get initial free memory
                        initial_free = torch.cuda.memory_reserved(
                            0
                        ) - torch.cuda.memory_allocated(0)
                        log_with_timestamp(
                            f"Initial free CUDA memory: {initial_free / 1024 / 1024:.1f}MB"
                        )

                    # Call the training function from util.py
                    util.train(
                        output_dir,
                        data_dir,
                        class_list,
                        learning_rate=float(learning_rate),
                        batch_size=int(batch_size),
                        iterations=int(iters_this_round),
                        checkpoint_period=int(checkpoint_period),
                        device=device,
                        model=model,
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e) or "CUDA out of memory" in str(e):
                        log_with_timestamp(
                            "Memory error occurred during util.train execution. Cleaning up..."
                        )
                        raise e  # Re-raise to be caught by outer exception handler
                    else:
                        raise e
                finally:
                    # Force garbage collection
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                end_time = time.time()

                # Calculate and log training time
                training_time = end_time - start_time
                log_with_timestamp(
                    f"Mini-batch training completed in {training_time:.2f} seconds"
                )

                # Update progress counters
                current_iter += iters_this_round
                remaining_iters -= iters_this_round

                log_with_timestamp(
                    f"Completed iterations up to {current_iter}/{iterations} ({(current_iter/iterations)*100:.1f}%)"
                )

                # Additional memory cleanup after mini-batch
                log_with_timestamp("Performing post-training memory cleanup...")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    log_with_timestamp("CUDA cache cleared")

                # Pause briefly to allow system resources to stabilize
                log_with_timestamp("Pausing briefly to stabilize system resources...")
                time.sleep(2)

            except RuntimeError as e:
                # Handle out-of-memory errors by reducing mini-batch size
                if "out of memory" in str(e) or "CUDA out of memory" in str(e):
                    log_with_timestamp(f"WARNING: Out of memory error: {str(e)}")
                    log_with_timestamp("Trying to reduce mini-batch size...")

                    if mini_batch_size > 1:
                        # Cut mini-batch size in half, but keep minimum of 1
                        mini_batch_size = max(1, mini_batch_size // 2)
                        log_with_timestamp(
                            f"Reduced mini-batch size to {mini_batch_size}, trying again..."
                        )
                        # Don't increment counters, retry with smaller mini-batch
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            log_with_timestamp("CUDA cache cleared")
                        log_with_timestamp(
                            "Waiting 5 seconds for memory to stabilize..."
                        )
                        time.sleep(5)  # Wait longer to ensure memory is cleared
                        continue
                    else:
                        # Can't reduce mini-batch size further
                        log_with_timestamp(
                            "ERROR: Already at minimum mini-batch size. Training cannot proceed."
                        )
                        log_with_timestamp(
                            "Try using a smaller model or increase system memory."
                        )
                        break
                else:
                    # For other runtime errors, propagate the exception
                    log_with_timestamp(f"ERROR: Runtime error occurred: {str(e)}")
                    raise e

        if current_iter == iterations:
            log_with_timestamp(
                f"=== TRAINING SUCCESSFULLY COMPLETED: {current_iter}/{iterations} iterations ==="
            )
        else:
            log_with_timestamp(
                f"=== TRAINING STOPPED EARLY: {current_iter}/{iterations} iterations completed ==="
            )

    except Exception as e:
        # Handle any other exceptions
        log_with_timestamp(f"CRITICAL ERROR during training: {str(e)}")
        log_with_timestamp("Traceback information:")
        import traceback

        traceback.print_exc()
        raise e
    finally:
        # Ensure memory monitoring thread stops when function exits
        log_with_timestamp("Shutting down memory monitoring...")
        exit_monitoring = True
        monitor_thread.join(
            timeout=1.0
        )  # Wait for thread to finish, but don't block indefinitely
        log_with_timestamp("=== TRAINING PROCESS ENDED ===")


if __name__ == "__main__":
    # Log startup information
    log_with_timestamp("Starting training script")

    # Create command-line argument parser
    parser = argparse.ArgumentParser(
        description="Train object detection models with minimal memory usage"
    )
    # Define command-line arguments with default values
    parser.add_argument(
        "--class-list",
        default="./class.names",
        help="Path to file containing class names",
    )
    parser.add_argument(
        "--data-dir", default="./data", help="Directory containing training data"
    )
    parser.add_argument(
        "--output-dir", default="./output", help="Directory to save model outputs"
    )
    parser.add_argument(
        "--device", default="cpu", help="Device to use for training (cpu or cuda)"
    )
    parser.add_argument(
        "--learning-rate", default=0.00005, help="Learning rate for optimizer"
    )
    parser.add_argument("--batch-size", default=1, help="Batch size for training")
    parser.add_argument(
        "--iterations", default=50, help="Total number of training iterations"
    )
    parser.add_argument(
        "--checkpoint-period", default=25, help="Save checkpoint every N iterations"
    )
    parser.add_argument(
        "--mini-batch-size",
        default=1,
        type=int,
        help="Number of iterations to train in each mini-batch",
    )
    parser.add_argument(
        "--model",
        default="COCO-Detection/retinanet_R_50_FPN_1x.yaml",
        help="Model configuration file",
    )

    # Parse arguments
    args = parser.parse_args()
    log_with_timestamp("Command line arguments parsed")

    # Log all arguments
    log_with_timestamp("Starting training with parameters:")
    for arg in vars(args):
        log_with_timestamp(f"  {arg}: {getattr(args, arg)}")

    # Start training with the specified arguments
    try:
        log_with_timestamp("Calling train_minibatches function")
        train_minibatches(
            args.output_dir,
            args.data_dir,
            args.class_list,
            device=args.device,
            learning_rate=float(args.learning_rate),
            batch_size=int(args.batch_size),
            iterations=int(args.iterations),
            checkpoint_period=int(args.checkpoint_period),
            model=args.model,
            mini_batch_size=args.mini_batch_size,
        )
    except KeyboardInterrupt:
        log_with_timestamp("Training interrupted by user (Ctrl+C)")
        sys.exit(1)
