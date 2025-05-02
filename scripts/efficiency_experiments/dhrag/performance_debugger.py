import logging
import time
import threading
import queue
import psutil
import os
from contextlib import contextmanager

class PerformanceDebugger:
    def __init__(self, log_interval=1.0, detailed_monitoring=False):
        self.log_interval = log_interval
        self.detailed_monitoring = detailed_monitoring
        self.monitoring = False
        self.metrics_queue = queue.Queue()
        self.monitor_thread = None
        self.start_time = None
        self.operation_times = {}
        self.current_operation = None
        
        # 设置日志
        logging.basicConfig(
            level=logging.DEBUG if detailed_monitoring else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def start_monitoring(self):
        """启动性能监控"""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logging.info("Performance monitoring started")
        if self.detailed_monitoring:
            logging.info("Detailed monitoring enabled")

    def stop_monitoring(self):
        """停止性能监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.analyze_results()
        logging.info("Performance monitoring stopped")

    def _monitor_resources(self):
        """监控系统资源使用情况"""
        while self.monitoring:
            try:
                process = psutil.Process(os.getpid())
                metrics = {
                    'timestamp': time.time(),
                    'cpu_percent': process.cpu_percent(),
                    'memory_rss': process.memory_info().rss / 1024 / 1024,  # MB
                    'memory_percent': process.memory_percent(),
                    'current_operation': self.current_operation
                }
                
                if self.detailed_monitoring:
                    # 添加更多详细指标
                    metrics.update({
                        'io_counters': process.io_counters()._asdict(),
                        'num_threads': process.num_threads(),
                        'num_fds': process.num_fds(),
                        'ctx_switches': process.num_ctx_switches()._asdict(),
                        'system_cpu_percent': psutil.cpu_percent(interval=None),
                        'system_memory': dict(psutil.virtual_memory()._asdict())
                    })
                    
                self.metrics_queue.put(metrics)
                time.sleep(self.log_interval)
            except Exception as e:
                logging.error(f"Error in resource monitoring: {str(e)}")

    @contextmanager
    def monitor_operation(self, operation_name):
        """监控特定操作的执行时间"""
        try:
            start_time = time.time()
            self.current_operation = operation_name
            if self.detailed_monitoring:
                logging.debug(f"Starting operation: {operation_name}")
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            if operation_name not in self.operation_times:
                self.operation_times[operation_name] = []
            self.operation_times[operation_name].append(duration)
            if self.detailed_monitoring:
                logging.debug(f"Completed operation: {operation_name} in {duration:.3f}s")
            self.current_operation = None

    def analyze_results(self):
        """分析性能监控结果"""
        if not self.metrics_queue.empty():
            metrics_list = []
            while not self.metrics_queue.empty():
                metrics_list.append(self.metrics_queue.get())

            # 基础统计
            stats = self._calculate_basic_stats(metrics_list)
            
            # 详细统计（如果启用）
            if self.detailed_monitoring:
                stats.update(self._calculate_detailed_stats(metrics_list))

            # 输出分析报告
            self._print_analysis(stats)
            self._save_results(stats)

    def _calculate_basic_stats(self, metrics_list):
        """计算基础统计指标"""
        return {
            'total_runtime': time.time() - self.start_time,
            'cpu_usage': {
                'mean': sum(m['cpu_percent'] for m in metrics_list) / len(metrics_list),
                'max': max(m['cpu_percent'] for m in metrics_list)
            },
            'memory_usage': {
                'mean': sum(m['memory_rss'] for m in metrics_list) / len(metrics_list),
                'max': max(m['memory_rss'] for m in metrics_list)
            }
        }

    def _calculate_detailed_stats(self, metrics_list):
        """计算详细统计指标"""
        if not self.detailed_monitoring:
            return {}
            
        return {
            'io_stats': {
                'read_bytes': sum(m['io_counters']['read_bytes'] for m in metrics_list if 'io_counters' in m),
                'write_bytes': sum(m['io_counters']['write_bytes'] for m in metrics_list if 'io_counters' in m)
            },
            'thread_stats': {
                'mean_threads': sum(m['num_threads'] for m in metrics_list if 'num_threads' in m) / len(metrics_list)
            },
            'system_stats': {
                'mean_system_cpu': sum(m['system_cpu_percent'] for m in metrics_list if 'system_cpu_percent' in m) / len(metrics_list)
            }
        }

    def _print_analysis(self, stats):
        """打印性能分析结果"""
        logging.info("\nPerformance Analysis Summary:")
        logging.info(f"Total Runtime: {stats['total_runtime']:.2f}s")
        logging.info(f"Average CPU Usage: {stats['cpu_usage']['mean']:.1f}%")
        logging.info(f"Peak CPU Usage: {stats['cpu_usage']['max']:.1f}%")
        logging.info(f"Average Memory Usage: {stats['memory_usage']['mean']:.1f} MB")
        logging.info(f"Peak Memory Usage: {stats['memory_usage']['max']:.1f} MB")

        if self.detailed_monitoring and 'io_stats' in stats:
            logging.info("\nDetailed Statistics:")
            logging.info(f"Total I/O Read: {stats['io_stats']['read_bytes'] / (1024*1024):.1f} MB")
            logging.info(f"Total I/O Write: {stats['io_stats']['write_bytes'] / (1024*1024):.1f} MB")
            logging.info(f"Average Thread Count: {stats['thread_stats']['mean_threads']:.1f}")
            logging.info(f"Average System CPU Usage: {stats['system_stats']['mean_system_cpu']:.1f}%")

        logging.info("\nOperation Timings:")
        for op_name, times in self.operation_times.items():
            avg_time = sum(times) / len(times)
            max_time = max(times)
            logging.info(f"{op_name}:")
            logging.info(f"  Average Time: {avg_time:.3f}s")
            logging.info(f"  Maximum Time: {max_time:.3f}s")
            if self.detailed_monitoring:
                logging.info(f"  Call Count: {len(times)}")

    def _save_results(self, stats):
        """保存性能分析结果到文件"""
        try:
            output_dir = "./output/performance_metrics"
            os.makedirs(output_dir, exist_ok=True)
            
            filename = os.path.join(output_dir, f'performance_report_{int(time.time())}.json')
            import json
            with open(filename, 'w') as f:
                json.dump(stats, f, indent=2)
            logging.info(f"\nPerformance report saved to: {filename}")
        except Exception as e:
            logging.error(f"Error saving performance results: {str(e)}")

    def _record_operation_time(self, operation_name, duration):
        """记录操作执行时间"""
        if operation_name not in self.operation_times:
            self.operation_times[operation_name] = []
        self.operation_times[operation_name].append(duration)
        if self.detailed_monitoring:
            logging.debug(f"{operation_name} completed in {duration:.3f}s")