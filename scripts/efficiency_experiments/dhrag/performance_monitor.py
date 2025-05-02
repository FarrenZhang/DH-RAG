import time
import psutil
import logging
import json
import os
from datetime import datetime
import numpy as np
import GPUtil
from threading import Thread
import queue

class PerformanceMonitor:
    def __init__(self, config):
        self.config = config
        self.metrics = {
            'retrieval_times': [],
            'generation_times': [],
            'total_times': [],
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'gpu_memory': []
        }
        self.start_time = None
        self.monitoring = False
        self.monitor_thread = None
        self.metric_queue = queue.Queue()
        
        # 创建性能监控输出目录
        self.performance_dir = os.path.join(config.OUTPUT_DIR, 'performance_metrics')
        os.makedirs(self.performance_dir, exist_ok=True)
        
        # 设置日志
        logging.basicConfig(
            filename=os.path.join(self.performance_dir, 'performance_monitor.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def start_monitoring(self):
        """开始系统资源监控"""
        self.monitoring = True
        self.monitor_thread = Thread(target=self._monitor_resources)
        self.monitor_thread.start()
        self.start_time = time.time()
        logging.info("Performance monitoring started")

    def stop_monitoring(self):
        """停止系统资源监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logging.info("Performance monitoring stopped")

    def _monitor_resources(self):
        """监控系统资源使用情况"""
        while self.monitoring:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            # GPU监控
            gpu_metrics = {'usage': 0, 'memory': 0}
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_metrics['usage'] = gpus[0].load * 100
                    gpu_metrics['memory'] = gpus[0].memoryUtil * 100
            except Exception as e:
                logging.warning(f"Unable to monitor GPU: {str(e)}")

            self.metric_queue.put({
                'timestamp': time.time(),
                'cpu_usage': cpu_percent,
                'memory_usage': memory_percent,
                'gpu_usage': gpu_metrics['usage'],
                'gpu_memory': gpu_metrics['memory']
            })
            time.sleep(1)

    def record_retrieval_time(self, time_taken):
        """记录检索时间"""
        self.metrics['retrieval_times'].append(time_taken)
        logging.info(f"Recorded retrieval time: {time_taken:.4f}s")

    def record_generation_time(self, time_taken):
        """记录生成时间"""
        self.metrics['generation_times'].append(time_taken)
        logging.info(f"Recorded generation time: {time_taken:.4f}s")

    def record_total_time(self, time_taken):
        """记录总处理时间"""
        self.metrics['total_times'].append(time_taken)
        logging.info(f"Recorded total processing time: {time_taken:.4f}s")

    def process_metrics(self):
        """处理并保存所有收集的指标"""
        # 处理队列中的系统资源数据
        while not self.metric_queue.empty():
            metric = self.metric_queue.get()
            self.metrics['cpu_usage'].append(metric['cpu_usage'])
            self.metrics['memory_usage'].append(metric['memory_usage'])
            self.metrics['gpu_usage'].append(metric['gpu_usage'])
            self.metrics['gpu_memory'].append(metric['gpu_memory'])

        # 计算统计数据
        stats = {
            'retrieval': self._calculate_stats(self.metrics['retrieval_times']),
            'generation': self._calculate_stats(self.metrics['generation_times']),
            'total': self._calculate_stats(self.metrics['total_times']),
            'cpu': self._calculate_stats(self.metrics['cpu_usage']),
            'memory': self._calculate_stats(self.metrics['memory_usage']),
            'gpu_usage': self._calculate_stats(self.metrics['gpu_usage']),
            'gpu_memory': self._calculate_stats(self.metrics['gpu_memory'])
        }

        # 保存原始数据
        self._save_raw_metrics()
        
        # 保存统计数据
        self._save_stats(stats)
        
        # 生成性能报告
        self._generate_report(stats)

        return stats

    def _calculate_stats(self, data):
        """计算统计指标"""
        if not data:
            return {
                'mean': 0,
                'median': 0,
                'std': 0,
                'min': 0,
                'max': 0,
                'p95': 0,
                'p99': 0
            }

        return {
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'p95': np.percentile(data, 95),
            'p99': np.percentile(data, 99)
        }

    def _save_raw_metrics(self):
        """保存原始指标数据"""
        filename = os.path.join(self.performance_dir, 'raw_metrics.json')
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        logging.info(f"Raw metrics saved to {filename}")

    def _save_stats(self, stats):
        """保存统计数据"""
        filename = os.path.join(self.performance_dir, 'performance_stats.json')
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=4)
        logging.info(f"Performance stats saved to {filename}")

    def _generate_report(self, stats):
        """生成性能报告"""
        report = f"""
Performance Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
=================================================================

1. Time Efficiency
-----------------
Retrieval Performance:
    Mean Time: {stats['retrieval']['mean']:.4f}s
    95th Percentile: {stats['retrieval']['p95']:.4f}s
    Maximum Time: {stats['retrieval']['max']:.4f}s

Generation Performance:
    Mean Time: {stats['generation']['mean']:.4f}s
    95th Percentile: {stats['generation']['p95']:.4f}s
    Maximum Time: {stats['generation']['max']:.4f}s

Total Processing Time:
    Mean Time: {stats['total']['mean']:.4f}s
    95th Percentile: {stats['total']['p95']:.4f}s
    Maximum Time: {stats['total']['max']:.4f}s

2. Resource Utilization
----------------------
CPU Usage:
    Average: {stats['cpu']['mean']:.2f}%
    Peak: {stats['cpu']['max']:.2f}%

Memory Usage:
    Average: {stats['memory']['mean']:.2f}%
    Peak: {stats['memory']['max']:.2f}%

GPU Usage:
    Average: {stats['gpu_usage']['mean']:.2f}%
    Peak: {stats['gpu_usage']['max']:.2f}%
    Memory Usage (Avg): {stats['gpu_memory']['mean']:.2f}%
    Memory Usage (Peak): {stats['gpu_memory']['max']:.2f}%

3. Performance Analysis
----------------------
- System shows {'stable' if stats['cpu']['std'] < 10 else 'variable'} performance
- Resource utilization is {'optimal' if stats['cpu']['mean'] < 70 else 'high'}
- Memory usage is {'acceptable' if stats['memory']['max'] < 80 else 'concerning'}
- GPU utilization is {'efficient' if stats['gpu_usage']['mean'] > 50 else 'underutilized'}

Recommendations:
---------------
1. {'Consider increasing batch size' if stats['gpu_usage']['mean'] < 50 else 'Batch size appears optimal'}
2. {'Memory optimization may be needed' if stats['memory']['max'] > 80 else 'Memory usage is acceptable'}
3. {'CPU bottleneck detected' if stats['cpu']['mean'] > 80 else 'CPU usage is acceptable'}
"""

        # 保存报告
        filename = os.path.join(self.performance_dir, 'performance_report.txt')
        with open(filename, 'w') as f:
            f.write(report)
        logging.info(f"Performance report generated and saved to {filename}")