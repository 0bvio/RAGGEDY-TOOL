import psutil
import torch
import subprocess
import os
import re
from typing import Dict, Any, Optional

class ResourceMonitor:
    def __init__(self):
        pass

    def get_system_stats(self) -> Dict[str, Any]:
        """Returns current CPU, RAM, and VRAM usage."""
        stats = {}
        
        # CPU
        stats["cpu_percent"] = psutil.cpu_percent(interval=None)
        
        # RAM
        ram = psutil.virtual_memory()
        stats["ram_total_gb"] = round(ram.total / (1024**3), 2)
        stats["ram_used_gb"] = round(ram.used / (1024**3), 2)
        stats["ram_free_gb"] = round(ram.available / (1024**3), 2)
        stats["ram_percent"] = ram.percent
        
        # VRAM
        stats["vram_available"] = False
        if torch.cuda.is_available():
            try:
                # Use nvidia-smi for more accurate system-wide VRAM info
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,nounits,noheader"],
                    capture_output=True, text=True, check=True
                )
                parts = result.stdout.strip().split(",")
                if len(parts) == 3:
                    total, used, free = map(int, [p.strip() for p in parts])
                    stats["vram_available"] = True
                    stats["vram_total_gb"] = round(total / 1024, 2)
                    stats["vram_used_gb"] = round(used / 1024, 2)
                    stats["vram_free_gb"] = round(free / 1024, 2)
                    stats["vram_percent"] = round((used / total) * 100, 1)
            except Exception:
                # Fallback to torch (less accurate for system-wide)
                stats["vram_available"] = True
                vram_total = torch.cuda.get_device_properties(0).total_memory
                vram_reserved = torch.cuda.memory_reserved(0)
                vram_allocated = torch.cuda.memory_allocated(0)
                stats["vram_total_gb"] = round(vram_total / (1024**3), 2)
                stats["vram_used_gb"] = round(vram_allocated / (1024**3), 2)
                stats["vram_free_gb"] = round((vram_total - vram_allocated) / (1024**3), 2)
                stats["vram_percent"] = round((vram_allocated / vram_total) * 100, 1)
                
        return stats

    def can_accommodate(self, estimated_gb: float, device: str = "cuda") -> bool:
        """Checks if there is enough room for a model of estimated_gb size."""
        stats = self.get_system_stats()
        buffer_gb = 1.0 # Keep 1GB safety buffer
        
        if device == "cuda" and stats["vram_available"]:
            return stats["vram_free_gb"] >= (estimated_gb + buffer_gb)
        else:
            # Fallback to RAM
            return stats["ram_free_gb"] >= (estimated_gb + buffer_gb)

    def estimate_model_size(self, filename: str) -> float:
        """Estimates GGUF model size in GB based on filename or file size."""
        # Simple heuristic: file size on disk is a good proxy for RAM/VRAM usage for GGUF
        from llm.manager import ModelManager
        manager = ModelManager()
        path = manager.get_model_path(filename)
        if os.path.exists(path):
            return os.path.getsize(path) / (1024**3)
        
        # If not downloaded yet, try to guess from name (e.g. 8B, 14B)
        match = re.search(r'(\d+)B', filename, re.IGNORECASE)
        if match:
            b_count = int(match.group(1))
            # Q4_K_M is roughly 0.6GB per 1B parameters
            return b_count * 0.7 
            
        return 4.0 # Default guess
