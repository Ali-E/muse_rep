#!/usr/bin/env python3
"""
GPU Monitor - Automatically run a program when GPUs become available.

Usage:
    python gpu_monitor.py --command "python train.py" --threshold 10 --memory-threshold 1000
    python gpu_monitor.py --command "./run_experiment.sh" --gpus 0,1,2 --threshold 5
"""

import subprocess
import time
import argparse
import sys
from datetime import datetime


def get_gpu_info():
    """Get GPU utilization and memory usage for all GPUs."""
    try:
        # Query GPU utilization and memory
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [p.strip() for p in line.split(',')]
                gpu_id = int(parts[0])
                utilization = float(parts[1])
                memory_used = float(parts[2])
                memory_total = float(parts[3])
                
                gpus.append({
                    'id': gpu_id,
                    'utilization': utilization,
                    'memory_used': memory_used,
                    'memory_total': memory_total,
                    'memory_free': memory_total - memory_used
                })
        
        return gpus
    
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")
        return None
    except Exception as e:
        print(f"Error parsing GPU info: {e}")
        return None


def check_gpus_available(gpus, gpu_ids, util_threshold, memory_threshold):
    """Check if specified GPUs are below utilization and memory thresholds."""
    if gpus is None:
        return False
    
    # Filter to only check specified GPUs
    gpus_to_check = [g for g in gpus if g['id'] in gpu_ids]
    
    if not gpus_to_check:
        print(f"Warning: No GPUs found matching IDs {gpu_ids}")
        return False
    
    # Check if all specified GPUs are below thresholds
    for gpu in gpus_to_check:
        if gpu['utilization'] > util_threshold:
            return False
        if gpu['memory_used'] > memory_threshold:
            return False
    
    return True


def print_gpu_status(gpus, gpu_ids):
    """Print current GPU status."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] GPU Status:")
    
    if gpus is None:
        print("  Unable to query GPUs")
        return
    
    for gpu in gpus:
        if gpu['id'] in gpu_ids:
            print(f"  GPU {gpu['id']}: {gpu['utilization']:.1f}% util, "
                  f"{gpu['memory_used']:.0f}/{gpu['memory_total']:.0f} MB memory "
                  f"({gpu['memory_free']:.0f} MB free)")


def run_command(command, use_shell=True):
    """Run the specified command."""
    print(f"\n{'='*60}")
    print(f"LAUNCHING PROGRAM: {command}")
    print(f"{'='*60}\n")
    
    try:
        if use_shell:
            # Run as shell command (allows for pipes, redirects, etc.)
            result = subprocess.run(command, shell=True)
        else:
            # Run as separate arguments
            result = subprocess.run(command.split())
        
        print(f"\n{'='*60}")
        print(f"Program completed with exit code: {result.returncode}")
        print(f"{'='*60}\n")
        
        return result.returncode
    
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        return -1
    except Exception as e:
        print(f"Error running command: {e}")
        return -1


def main():
    parser = argparse.ArgumentParser(
        description='Monitor GPUs and run a program when they become available',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run when any GPU is below 10% utilization
  python gpu_monitor.py --command "python train.py"
  
  # Run when GPUs 0 and 1 are both below 5% utilization
  python gpu_monitor.py --command "./experiment.sh" --gpus 0,1 --threshold 5
  
  # Check utilization and memory
  python gpu_monitor.py --command "python train.py" --threshold 10 --memory-threshold 2000
  
  # Run once and exit
  python gpu_monitor.py --command "python train.py" --run-once
        """
    )
    
    parser.add_argument(
        '--command', '-c',
        type=str,
        required=True,
        help='Command to run when GPUs are available'
    )
    
    parser.add_argument(
        '--gpus', '-g',
        type=str,
        default=None,
        help='Comma-separated list of GPU IDs to monitor (default: all GPUs)'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=10.0,
        help='GPU utilization threshold in percent (default: 10.0)'
    )
    
    parser.add_argument(
        '--memory-threshold', '-m',
        type=float,
        default=1000.0,
        help='GPU memory usage threshold in MB (default: 1000.0)'
    )
    
    parser.add_argument(
        '--interval', '-i',
        type=float,
        default=2.0,
        help='Polling interval in seconds (default: 2.0)'
    )
    
    parser.add_argument(
        '--run-once',
        action='store_true',
        help='Run the command once and exit (do not continue monitoring)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print GPU status on every check'
    )
    
    args = parser.parse_args()
    
    # Parse GPU IDs
    if args.gpus is None:
        # Get all available GPUs
        initial_gpus = get_gpu_info()
        if initial_gpus is None:
            print("Error: Could not detect GPUs")
            sys.exit(1)
        gpu_ids = [g['id'] for g in initial_gpus]
    else:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    
    print("="*60)
    print("GPU Monitor Started")
    print("="*60)
    print(f"Monitoring GPUs: {gpu_ids}")
    print(f"Utilization threshold: {args.threshold}%")
    print(f"Memory threshold: {args.memory_threshold} MB")
    print(f"Check interval: {args.interval} seconds")
    print(f"Command: {args.command}")
    print(f"Run once: {args.run_once}")
    print("="*60)
    print("\nWaiting for GPUs to become available...")
    print("(Press Ctrl+C to stop monitoring)\n")
    
    try:
        while True:
            gpus = get_gpu_info()
            
            if args.verbose:
                print_gpu_status(gpus, gpu_ids)
            
            if check_gpus_available(gpus, gpu_ids, args.threshold, args.memory_threshold):
                print_gpu_status(gpus, gpu_ids)
                print(f"\n✓ GPUs {gpu_ids} are available!")
                
                exit_code = run_command(args.command)
                
                if args.run_once:
                    print("Run-once mode: Exiting")
                    sys.exit(exit_code)
                else:
                    print("Continuing to monitor GPUs...")
                    print("(The program may be scheduled to run again if GPUs become available)\n")
            
            time.sleep(args.interval)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
        sys.exit(0)


if __name__ == '__main__':
    main()
