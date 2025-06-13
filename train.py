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

# Global flag to signal when memory monitoring should stop
exit_monitoring = False


def monitor_memory_usage():
    """
    Monitor and print memory usage of the current process and system.
    Runs in a separate thread and updates every 10 seconds.
    """
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
        print(
            f"[Memory Monitor] Process: {memory_gb:.2f}GB | System used: {system_memory_used_percent}% | Available: {system_memory_available_gb:.2f}GB"
        )
        time.sleep(10)  # Wait 10 seconds before next update


def train_minibatches(
    output_dir,  # Directory to save model checkpoints
    data_dir,  # Directory containing training data
    class_list,  # File containing class names
    device="cpu",  # Device to use for training (cpu or cuda)
    learning_rate=0.00005,  # Learning rate for optimizer
    batch_size=1,  # Number of samples per batch
    iterations=50,  # Total number of training iterations
    checkpoint_period=25,  # Save model every this many iterations
    model="COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml",  # Model configuration
    mini_batch_size=5,  # Number of iterations to run before freeing memory
):
    """
    Train a detection model using very small batches to minimize memory usage.
    This is especially useful for machines with limited memory or for large models.

    This function breaks down training into mini-batches of iterations, performing
    memory cleanup between each mini-batch to prevent out-of-memory errors.
    """
    # Clean up memory before starting training
    gc.collect()
    # Clear GPU memory if available
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Start memory monitoring in a separate thread
    global exit_monitoring
    exit_monitoring = False
    monitor_thread = Thread(target=monitor_memory_usage)
    monitor_thread.daemon = True  # Thread will exit when main program exits
    monitor_thread.start()

    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Print training configuration information
        print(f"Starting ultra-low memory training:")
        print(f"  - Device: {device}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Iterations: {iterations}")
        print(f"  - Training in mini-batches of {mini_batch_size} iterations")
        print(f"  - Model: {model}")

        # Initialize counters for the training loop
        remaining_iters = iterations
        current_iter = 0

        # Continue training until all iterations are completed
        while remaining_iters > 0:
            # Calculate how many iterations to run in this mini-batch
            iters_this_round = min(mini_batch_size, remaining_iters)
            print(
                f"\n===== Training mini-batch {current_iter} to {current_iter + iters_this_round} of {iterations} ====="
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

                print(f"Completed iterations up to {current_iter}/{iterations}")

                # Additional memory cleanup after mini-batch
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

                # Pause briefly to allow system resources to stabilize
                time.sleep(2)

            except RuntimeError as e:
                # Handle out-of-memory errors by reducing mini-batch size
                if "out of memory" in str(e) or "CUDA out of memory" in str(e):
                    print(f"WARNING: Out of memory error: {str(e)}")
                    print("Trying to reduce mini-batch size...")

                    if mini_batch_size > 1:
                        # Cut mini-batch size in half, but keep minimum of 1
                        mini_batch_size = max(1, mini_batch_size // 2)
                        print(
                            f"Reduced mini-batch size to {mini_batch_size}, trying again..."
                        )
                        # Don't increment counters, retry with smaller mini-batch
                        gc.collect()
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        time.sleep(5)  # Wait longer to ensure memory is cleared
                        continue
                    else:
                        # Can't reduce mini-batch size further
                        print(
                            "ERROR: Already at minimum mini-batch size. Training cannot proceed."
                        )
                        print("Try using a smaller model or increase system memory.")
                        break
                else:
                    # For other runtime errors, propagate the exception
                    print(f"ERROR: {str(e)}")
                    raise e

        print(f"Training completed: {current_iter}/{iterations} iterations")

    except Exception as e:
        # Handle any other exceptions
        print(f"Error during training: {str(e)}")
        raise e
    finally:
        # Ensure memory monitoring thread stops when function exits
        exit_monitoring = True
        monitor_thread.join(
            timeout=1.0
        )  # Wait for thread to finish, but don't block indefinitely


if __name__ == "__main__":
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
        default=5,
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

    # Start training with the specified arguments
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
