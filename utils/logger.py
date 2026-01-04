import logging
import os
import sys
from datetime import datetime

class Logger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.current_log_file = os.path.join(log_dir, f"raggedy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # Setup logging to file and console
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.FileHandler(self.current_log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("RAGGEDY")

    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def error(self, msg):
        self.logger.error(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def get_logs(self):
        if os.path.exists(self.current_log_file):
            with open(self.current_log_file, 'r') as f:
                return f.read()
        return ""

    def get_all_log_files(self):
        return sorted([f for f in os.listdir(self.log_dir) if f.endswith(".log")], reverse=True)

    def read_log_file(self, filename):
        path = os.path.join(self.log_dir, filename)
        if os.path.exists(path):
            with open(path, 'r') as f:
                return f.read()
        return ""

# Global instance
raggedy_logger = Logger()
