#!/usr/bin/env python3
"""
Power monitoring script using tegrastats for Jetson devices.
Collects VDD_IN, VDD_CPU_GPU_CV, and VDD_SOC statistics.
"""

import subprocess
import re
import time
import signal
import sys
import statistics
import argparse
from collections import defaultdict
from datetime import datetime

class TegrastatsMonitor:
    def __init__(self, output_file="power_stats.txt"):
        self.running = False
        self.stats = defaultdict(list)
        self.start_time = None
        self.end_time = None
        self.output_file = output_file
        
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\nStopping tegrastats monitoring...")
        self.running = False
        
    def parse_tegrastats_line(self, line):
        """Parse a tegrastats output line and extract power values"""
        # Pattern to match power values: VDD_IN 6693mW/6693mW VDD_CPU_GPU_CV 1065mW/1065mW VDD_SOC 1825mW/1825mW
        pattern = r'VDD_IN (\d+)mW/(\d+)mW VDD_CPU_GPU_CV (\d+)mW/(\d+)mW VDD_SOC (\d+)mW/(\d+)mW'
        match = re.search(pattern, line)
        
        if match:
            vdd_in_current, vdd_in_max = int(match.group(1)), int(match.group(2))
            vdd_cpu_gpu_cv_current, vdd_cpu_gpu_cv_max = int(match.group(3)), int(match.group(4))
            vdd_soc_current, vdd_soc_max = int(match.group(5)), int(match.group(6))
            
            return {
                'VDD_IN': vdd_in_current,
                'VDD_CPU_GPU_CV': vdd_cpu_gpu_cv_current,
                'VDD_SOC': vdd_soc_current
            }
        return None
    
    def collect_stats(self):
        """Start tegrastats and collect power statistics"""
        print("Starting tegrastats monitoring...")
        print("Press Ctrl+C to stop and generate statistics")
        print("-" * 60)
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
        try:
            # Start tegrastats process
            process = subprocess.Popen(
                ['tegrastats', '--interval', '100'],  # 100ms interval
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            
            self.running = True
            self.start_time = time.time()
            
            # Read tegrastats output
            while self.running:
                line = process.stdout.readline()
                if not line:
                    break
                    
                # Parse the line for power statistics
                power_data = self.parse_tegrastats_line(line)
                if power_data:
                    for key, value in power_data.items():
                        self.stats[key].append(value)
                    
                    # Print current values for monitoring
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    print(f"[{timestamp}] VDD_IN: {power_data['VDD_IN']}mW, "
                          f"VDD_CPU_GPU_CV: {power_data['VDD_CPU_GPU_CV']}mW, "
                          f"VDD_SOC: {power_data['VDD_SOC']}mW")
                
                time.sleep(0.001)  # Small delay to prevent excessive CPU usage
            
            self.end_time = time.time()
            
        except Exception as e:
            print(f"Error during monitoring: {e}")
        finally:
            if 'process' in locals():
                process.terminate()
                process.wait()
    
    def calculate_statistics(self):
        """Calculate and display statistics for collected data"""
        if not any(self.stats.values()):
            print("No data collected!")
            return
        
        print("\n" + "=" * 60)
        print("POWER STATISTICS SUMMARY")
        print("=" * 60)
        
        duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        print(f"Monitoring Duration: {duration:.2f} seconds")
        print(f"Data Points Collected: {len(list(self.stats.values())[0])}")
        print()
        
        for power_type, values in self.stats.items():
            if values:
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0
                min_val = min(values)
                max_val = max(values)
                
                print(f"{power_type}:")
                print(f"  Mean: {mean_val:.2f} mW")
                print(f"  Std Dev: {std_val:.2f} mW")
                print(f"  Min: {min_val} mW")
                print(f"  Max: {max_val} mW")
                print(f"  Samples: {len(values)}")
                print()
    
    def save_to_file(self):
        """Save statistics to a file"""
        if not any(self.stats.values()):
            print("No data to save!")
            return
        
        with open(self.output_file, 'w') as f:
            f.write("POWER MONITORING STATISTICS\n")
            f.write("=" * 40 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {self.end_time - self.start_time:.2f} seconds\n")
            f.write(f"Data Points: {len(list(self.stats.values())[0])}\n\n")
            
            for power_type, values in self.stats.items():
                if values:
                    mean_val = statistics.mean(values)
                    std_val = statistics.stdev(values) if len(values) > 1 else 0
                    min_val = min(values)
                    max_val = max(values)
                    
                    f.write(f"{power_type}:\n")
                    f.write(f"  Mean: {mean_val:.2f} mW\n")
                    f.write(f"  Std Dev: {std_val:.2f} mW\n")
                    f.write(f"  Min: {min_val} mW\n")
                    f.write(f"  Max: {max_val} mW\n")
                    f.write(f"  Samples: {len(values)}\n\n")
        
        print(f"Statistics saved to {self.output_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Monitor power consumption using tegrastats')
    parser.add_argument('-o', '--output', default='power_stats.txt',
                       help='Output file name for statistics (default: power_stats.txt)')
    
    args = parser.parse_args()
    
    print("Tegrastats Power Monitor")
    print("=" * 30)
    print("This script will monitor power consumption using tegrastats.")
    print("Press Ctrl+C to stop monitoring")
    print(f"Results will be saved to: {args.output}")
    print()
    
    monitor = TegrastatsMonitor(args.output)
    
    try:
        monitor.collect_stats()
        monitor.calculate_statistics()
        monitor.save_to_file()
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
        monitor.calculate_statistics()
        monitor.save_to_file()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
