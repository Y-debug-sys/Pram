import os
import time
import torch
import logging
import threading

from queue import Queue
from enum import Enum, unique
from torch.utils.tensorboard import SummaryWriter
from logging.handlers import QueueHandler, QueueListener


def ensure_dir(path):
    """
    Ensures that the specified path exists, creates the directory if it does not exist
    
    Args:
        path (str): The directory path that needs to be ensured
    
    Returns:
        bool: Returns True if the directory already existed before the call, otherwise returns False
    """
    flag = os.path.exists(path)
    os.makedirs(path, exist_ok=True)
    return flag


class ExperimentLoggingManager:
    """
    A singleton class that manages logging, TensorBoard writers, and model saving
    for experiments. It ensures centralized and thread-safe handling of logs, metrics, and model checkpoints,
    organized according to specified directory modes.
    """

    _instance = None
    _lock = threading.Lock()

    @unique
    class LOG_DIR_MODE(Enum):
        """Enumeration for different log directory organization modes."""
        DATE_FIRST = 0      # Organizes logs by date first, then by run number
        NUMBER_FIRST = 1    # Organizes logs by run number first
        CATEGORY_FIRST = 2  # Organizes logs by category first
    
    # Root directory mapping for different log directory modes
    LOG_DIR_MODE_ROOT_DICT = {
        LOG_DIR_MODE.DATE_FIRST: 'log', 
        LOG_DIR_MODE.NUMBER_FIRST: 'log', 
        LOG_DIR_MODE.CATEGORY_FIRST: ''
    }

    # Default directory mode
    DIR_MODE = LOG_DIR_MODE.DATE_FIRST

    def __new__(cls, log_root=None):
        """
        Implements singleton pattern with thread safety using double-checked locking.
        
        Args:
            log_root (str, optional): Root directory for logs. Uses default if None.
            
        Returns:
            ExperimentLoggingManager: Instance of the logging manager
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ExperimentLoggingManager, cls).__new__(cls)
                if log_root is None:
                    log_root = cls.LOG_DIR_MODE_ROOT_DICT[cls.DIR_MODE]
                cls._instance._initialize(log_root, cls.DIR_MODE)
            return cls._instance

    def _initialize(self, log_root, dir_mode):
        """
        Initializes the logging manager with specified directory mode.
        
        Args:
            log_root (str): Root directory for logs
            dir_mode (LOG_DIR_MODE): Directory organization mode
        """
        # Initialize the queue for thread-safe logging
        self.log_queue = Queue(-1)

        # Set up logging directories based on the selected mode
        if dir_mode == self.LOG_DIR_MODE.DATE_FIRST:
            # Format: log/YYYYMMDD/runN/
            log_root = os.path.join(log_root, time.strftime("experiment_%Y%m%d"))
            run_num = 1
            while ensure_dir(os.path.join(log_root, f'run{run_num}')):
                run_num += 1
            self.log_dir = os.path.join(log_root, f'run{run_num}')
            self.log_file_path = os.path.join(self.log_dir,
                                              f'experiment_{time.strftime("%Y.%m.%d_%H:%M")}_PID{os.getpid()}_PPID{threading.get_ident()}.log')
            self.tb_dir_path = self.log_dir
            self.model_dir_path = os.path.join(self.log_dir, 'models')
            ensure_dir(self.model_dir_path)
            self.result_dir_path = os.path.join(self.log_dir, 'results')
            ensure_dir(self.result_dir_path)
        elif dir_mode == self.LOG_DIR_MODE.NUMBER_FIRST:
            # Format: log/runN/
            run_num = 1
            while ensure_dir(os.path.join(log_root, f'run{run_num}')):
                run_num += 1
            self.log_dir = os.path.join(log_root, f'run{run_num}')
            self.log_file_path = os.path.join(self.log_dir, f'experiment_{time.strftime("%Y.%m.%d_%H:%M")}_PID{os.getpid()}_PPID{threading.get_ident()}.log')
            self.tb_dir_path = self.log_dir
            self.model_dir_path = os.path.join(self.log_dir, 'models')
            ensure_dir(self.model_dir_path)
            self.result_dir_path = os.path.join(self.log_dir, 'results')
            ensure_dir(self.result_dir_path)
        elif dir_mode == self.LOG_DIR_MODE.CATEGORY_FIRST:
            # Format: category/logs/, tbs/, models/
            self.log_dir = log_root
            self.log_file_path = os.path.join(self.log_dir, 'logs', f'experiment_{time.strftime("%Y.%m.%d_%H:%M")}_PID{os.getpid()}_PPID{threading.get_ident()}.log')
            self.tb_dir_path = os.path.join(self.log_dir, 'tbs', f'experiment_{time.strftime("%Y.%m.%d_%H:%M")}')
            self.model_dir_path = os.path.join(self.log_dir, 'models', f'experiment_{time.strftime("%Y.%m.%d_%H:%M")}')
            self.result_dir_path = os.path.join(self.log_dir, 'results', f'experiment_{time.strftime("%Y.%m.%d_%H:%M")}')
            ensure_dir(os.path.join(self.log_dir, 'logs'))
            ensure_dir(self.tb_dir_path)
            ensure_dir(self.model_dir_path)
            ensure_dir(self.result_dir_path)

        # Set up file handler for persistent logging
        file_handler = logging.FileHandler(self.log_file_path)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        # Create and start queue listener for thread-safe logging
        self.queue_listener = QueueListener(self.log_queue, file_handler, respect_handler_level=True)
        self.queue_listener.start()

        # Initialize collections to store loggers, writers, and savers
        self.loggers = {}
        self.writers = {}
        self.model_savers = {}

    class ExperimentLogger:
        """
        Wrapper for standard logging module providing structured logging functionality
        with file and console outputs.
        """
        
        def __init__(self, logger_name, log_file_path, log_queue):
            """
            Initialize the experiment logger with name, file path and queue.
            
            Args:
                logger_name (str): Name of the logger
                log_file_path (str): Path to the log file
                log_queue (Queue): Queue for thread-safe logging
            """
            self.experiment_name = logger_name
            self.logger = logging.getLogger(logger_name)
            self.logger.setLevel(logging.INFO)
            # Add queue handler for thread-safe logging
            self.logger.addHandler(QueueHandler(log_queue))

            # Add file handler for persistent logging
            file_handler = logging.FileHandler(log_file_path)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            # Add stream handler for console output
            stream_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

        def log(self, message, level=logging.INFO):
            """
            Log a message with the specified logging level.
            
            Args:
                message (str): Message to log
                level (int): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            """
            if level == logging.DEBUG:
                self.logger.debug(message)
            elif level == logging.INFO:
                self.logger.info(message)
            elif level == logging.WARNING:
                self.logger.warning(message)
            elif level == logging.ERROR:
                self.logger.error(message)
            elif level == logging.CRITICAL:
                self.logger.critical(message)

        def close(self):
            """
            Close all handlers associated with this logger to free resources.
            """
            handlers = self.logger.handlers[:]
            for handler in handlers:
                handler.close()
                self.logger.removeHandler(handler)

    class ExperimentTBWriter:
        """
        Wrapper for TensorBoard SummaryWriter to manage metric logging.
        """
        
        def __init__(self, writer_name, tb_dir_path):
            """
            Initialize TensorBoard writer.
            
            Args:
                writer_name (str): Name of the writer
                tb_dir_path (str): Directory path for TensorBoard logs
            """
            self.writer_name = writer_name
            self.writer = SummaryWriter(tb_dir_path)

        def log_metric(self, tag, value, step):
            """
            Log a scalar metric to TensorBoard.
            
            Args:
                tag (str): Tag name for the metric
                value (float): Metric value
                step (int): Step number for the metric
            """
            self.writer.add_scalar(tag, value, step)

        def log_metrics(self, main_tag, tag_value_dict, step):
            """
            Log multiple metrics under a main tag.
            
            Args:
                main_tag (str): Main tag name grouping the metrics
                tag_value_dict (dict): Dictionary mapping tag names to values
                step (int): Step number for the metrics
            """
            self.writer.add_scalars(main_tag, tag_value_dict, step)

        def close(self):
            """
            Close the TensorBoard writer to flush remaining data and free resources.
            """
            self.writer.close()

    # class ModelSaver:
    #     """
    #     Handles saving models with automatic tracking of best/worst performing models.
    #     """
        
    #     def __init__(self, model_name, model_dir_path):
    #         """
    #         Initialize model saver.
            
    #         Args:
    #             model_name (str): Base name for the model
    #             model_dir_path (str): Directory path to save models
    #         """
    #         self.model_name = model_name
    #         self.model_dir_path = model_dir_path
    #         self.model_metric_list = []
    #         self.highest_metric_index = -1
    #         self.lowest_metric_index = -1

    #     def save(self, model, total_step, metric, epoch=None, episode=None, step=None):
    #         """
    #         Save the model with metadata in the filename.
            
    #         Args:
    #             model: PyTorch model to save
    #             total_step (int): Total training steps
    #             metric (float): Performance metric for the model
    #             epoch (int, optional): Current epoch number
    #             episode (int, optional): Current episode number
    #             step (int, optional): Current step number
    #         """
    #         model_name = f'{self.model_name}_total_step{total_step}'
    #         if epoch is not None:
    #             model_name += f'_epoch{epoch}'
    #         if episode is not None:
    #             model_name += f'_episode{episode}'
    #         if step is not None:
    #             model_name += f'_step{step}'
            
    #         # Track best and worst performing models based on metric
    #         self.model_metric_list.append((model_name, metric, (total_step, epoch, step)))
    #         current_idx = len(self.model_metric_list) - 1
            
    #         # Update highest metric index if current metric is better
    #         if self.highest_metric_index == -1 or metric > self.model_metric_list[self.highest_metric_index][1]:
    #             self.highest_metric_index = current_idx
    #         # Update lowest metric index if current metric is worse
    #         if self.lowest_metric_index == -1 or metric < self.model_metric_list[self.lowest_metric_index][1]:
    #             self.lowest_metric_index = current_idx
                
    #         model_path = os.path.join(self.model_dir_path, f'{model_name}.pth')
    #         torch.save(model, model_path)

    #     def get_highest_metric(self):
    #         """
    #         Get the model with the highest recorded metric.
            
    #         Returns:
    #             tuple: (best_model_path, model_name, metric_value)
    #         """
    #         best_model_name, best_metric, _ = self.model_metric_list[self.highest_metric_index]
    #         best_model_path = os.path.join(self.model_dir_path, f'{best_model_name}.pth')
    #         return torch.load(best_model_path), best_model_name, best_metric

    #     def get_lowest_metric(self):
    #         """
    #         Get the model with the lowest recorded metric.
            
    #         Returns:
    #             tuple: (best_model_path, model_name, metric_value)
    #         """
    #         best_model_name, best_metric, _ = self.model_metric_list[self.lowest_metric_index]
    #         best_model_path = os.path.join(self.model_dir_path, f'{best_model_name}.pth')
    #         return torch.load(best_model_path), best_model_name, best_metric

    def get_logger(self, logger_name):
        """
        Retrieve or create an ExperimentLogger instance by name.
        
        Args:
            logger_name (str): Name of the logger to retrieve/create
            
        Returns:
            ExperimentLogger: Logger instance
        """
        if logger_name not in self.loggers:
            self.loggers[logger_name] = self.ExperimentLogger(logger_name, self.log_file_path, self.log_queue)
        return self.loggers[logger_name]

    def get_writer(self, writer_name):
        """
        Retrieve or create an ExperimentTBWriter instance by name.
        This method can only be called from the main thread due to TensorBoard constraints.
        
        Args:
            writer_name (str): Name of the writer to retrieve/create
            
        Returns:
            ExperimentTBWriter: Writer instance
        """
        # Ensure this is only called from the main thread
        assert threading.get_ident() == threading.main_thread().ident, "SummaryWriter is only allowed in the main thread."
        if writer_name not in self.writers:
            self.writers[writer_name] = self.ExperimentTBWriter(writer_name, self.tb_dir_path)
        return self.writers[writer_name]

    # def get_model_saver(self, model_name):
    #     """
    #     Retrieve or create a ModelSaver instance by name.
        
    #     Args:
    #         model_name (str): Name of the model saver to retrieve/create
            
    #     Returns:
    #         ModelSaver: Model saver instance
    #     """
    #     if model_name not in self.model_savers:
    #         self.model_savers[model_name] = self.ModelSaver(model_name, self.model_dir_path)
    #     return self.model_savers[model_name]

    def get_model_directory(self):
        """
        Get the directory path for saving models.
        
        Returns:
            str: Directory path for saving models
        """
        return self.model_dir_path
    
    def get_result_directory(self):
        """
        Get the directory path for saving results.
        
        Returns:
            str: Directory path for saving results
        """
        return self.result_dir_path

    def close_all(self):
        """
        Close all resources to prevent resource leaks.
        """
        self.queue_listener.stop()
        for logger in self.loggers.values():
            logger.close()
        for writer in self.writers.values():
            writer.close()
