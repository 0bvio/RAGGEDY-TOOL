import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

class ChatAuditor:
    def __init__(self, log_dir: str = "logs/audit"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.audit_file = os.path.join(self.log_dir, "chat_audit.jsonl")

    def log_interaction(self, event_type: str, data: Dict[str, Any], chat_id: Optional[str] = None):
        """
        Logs a chat interaction event.
        event_type: 'request', 'response', 'chunk', etc.
        data: The content to log
        chat_id: Optional chat ID for per-chat logging
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }
        if chat_id:
            entry["chat_id"] = chat_id

        # 1. Log to global audit file
        try:
            with open(self.audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"FAILED TO GLOBAL AUDIT LOG: {e}")

        # 2. Log to per-chat audit file if chat_id is provided
        if chat_id:
            chat_log_file = os.path.join(self.log_dir, f"chat_{chat_id}.jsonl")
            try:
                with open(chat_log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception as e:
                print(f"FAILED TO PER-CHAT AUDIT LOG: {e}")

# Global instance
chat_auditor = ChatAuditor()
