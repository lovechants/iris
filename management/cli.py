import argparse
import sys
from management.partition_util import partitioner, print_device_info

def create_parser():
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Metal Runtime for GPU Computing",
        prog="iris"
    )
    
    parser.add_argument(
        "--info", 
        action="store_true",
        help="Display GPU information"
    )
    
    parser.add_argument(
        "--recommendation", 
        action="store_true",
        help="Display performance recommendations based on your machine"
    )
    
    return parser

def handle_info():
    """Handle the --info option."""
    print("DEBUG: handle_info() called")
    print("GPU Device Information:")
    try:
        print_device_info()
    except Exception as e:
        print(f"Error getting GPU information: {e}")
        sys.exit(1)

def handle_recommendation():
    """Handle the --recommendation option."""
    print("Performance Recommendations:") 
    try:
        props = partitioner.gpu_properties
        
        print(f"For your {props['name']} GPU:")
        print()
        
        thread_width = props['thread_execution_width']
        print(f"• Thread Execution Width: {thread_width}")
        print(f"  - Use block sizes that are multiples of {thread_width} for optimal performance")
        print(f"  - Examples: {thread_width}, {thread_width*2}, {thread_width*4}, etc.")
        print()
        
        max_threads = props['max_threads_per_threadgroup']
        print(f"• Max Threads per Threadgroup: {max_threads}")
        print(f"  - Avoid exceeding this limit in your block sizes")
        print()
        
        memory_size = props['threadgroup_memory_size']
        print(f"• Threadgroup Memory Size: {memory_size} bytes")
        print(f"  - Be mindful of shared memory usage per thread")
        print(f"  - Each thread can use up to ~{memory_size // max_threads} bytes if fully utilizing the threadgroup")
        print()
        
        if 'max_buffer_length' in props:
            max_buffer = props['max_buffer_length']
            print(f"• Max Buffer Length: {max_buffer:,} bytes ({max_buffer // (1024*1024):.1f} MB)")
            print(f"  - Large buffers may need to be split into chunks")
            print()
        
        print("Example Configurations:")
        print("• For small arrays (< 1000 elements):")
        small_grid, small_block = partitioner.partition_1d(512)
        print(f"  - Grid: {small_grid}, Block: {small_block}")
        print()
        
        print("• For medium arrays (1000-10000 elements):")
        med_grid, med_block = partitioner.partition_1d(5000)
        print(f"  - Grid: {med_grid}, Block: {med_block}")
        print()
        
        print("• For large arrays (> 10000 elements):")
        large_grid, large_block = partitioner.partition_1d(50000)
        print(f"  - Grid: {large_grid}, Block: {large_block}")
        print()
        
        if 'Apple' in props['name']:
            print("Apple Silicon Specific Tips:")
            print("• Avoid branching when possible")
            print("• Use vector types (float4, int4) when processing multiple values")
            print("• Consider memory coalescing for better performance")
        
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        sys.exit(1)

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.info and not args.recommendation:
        parser.print_help()
        return
    
    if args.info:
        handle_info()
    
    if args.recommendation:
        handle_recommendation()
