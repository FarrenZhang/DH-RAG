import time
import functools
import logging

def timer_decorator(monitor):
    """用于测量函数执行时间的装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            time_taken = end_time - start_time
            
            # 根据函数名称记录不同类型的时间
            if 'retrieve' in func.__name__:
                monitor.record_retrieval_time(time_taken)
            elif 'generate' in func.__name__:
                monitor.record_generation_time(time_taken)
            
            return result
        return wrapper
    return decorator

class BatchSizeOptimizer:
    """批处理大小优化器"""
    def __init__(self, initial_batch_size=32):
        self.current_batch_size = initial_batch_size
        self.performance_history = []
        
    def update(self, processing_time, memory_usage):
        """更新批处理大小"""
        self.performance_history.append({
            'batch_size': self.current_batch_size,
            'time': processing_time,
            'memory': memory_usage
        })
        
        # 基于最近的性能历史调整批大小
        if len(self.performance_history) >= 3:
            recent_perf = self.performance_history[-3:]
            avg_time = sum(p['time'] for p in recent_perf) / 3
            avg_memory = sum(p['memory'] for p in recent_perf) / 3
            
            if avg_memory < 70 and avg_time < 1.0:  # 如果内存使用率低且处理快
                self.current_batch_size = min(self.current_batch_size * 2, 256)
            elif avg_memory > 90 or avg_time > 2.0:  # 如果内存使用率高或处理慢
                self.current_batch_size = max(self.current_batch_size // 2, 16)
                
        return self.current_batch_size

class ResourceMonitor:
    """资源使用监控器"""
    def __init__(self):
        self.warning_thresholds = {
            'cpu': 80,
            'memory': 85,
            'gpu': 90
        }
        
    def check_resources(self, metrics):
        """检查资源使用情况并发出警告"""
        warnings = []
        
        if metrics['cpu_usage'][-1] > self.warning_thresholds['cpu']:
            warnings.append(f"High CPU usage: {metrics['cpu_usage'][-1]}%")
            
        if metrics['memory_usage'][-1] > self.warning_thresholds['memory']:
            warnings.append(f"High memory usage: {metrics['memory_usage'][-1]}%")
            
        if metrics['gpu_usage'][-1] > self.warning_thresholds['gpu']:
            warnings.append(f"High GPU usage: {metrics['gpu_usage'][-1]}%")
            
        for warning in warnings:
            logging.warning(warning)
            
        return warnings

def load_test_generator(num_queries, query_template):
    """生成负载测试查询"""
    for i in range(num_queries):
        yield query_template.format(i=i)