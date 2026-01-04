import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional

class ChatManager:
    def __init__(self, data_root: str = "data"):
        self.chats_dir = os.path.join(data_root, "chats")
        os.makedirs(self.chats_dir, exist_ok=True)
        self.active_chat_id = None

    def list_chats(self) -> List[Dict]:
        chats = []
        for filename in os.listdir(self.chats_dir):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(self.chats_dir, filename), 'r') as f:
                        chat_data = json.load(f)
                        chats.append({
                            "id": chat_data["id"],
                            "title": chat_data.get("title", "New Chat"),
                            "updated_at": chat_data.get("updated_at", "")
                        })
                except:
                    continue
        return sorted(chats, key=lambda x: x["updated_at"], reverse=True)

    def create_chat(self, title: str = "New Chat") -> str:
        chat_id = str(uuid.uuid4())
        chat_data = {
            "id": chat_id,
            "title": title,
            "messages": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        self._save_chat(chat_data)
        return chat_id

    def load_chat(self, chat_id: str) -> Optional[Dict]:
        path = os.path.join(self.chats_dir, f"{chat_id}.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return None

    def save_chat(self, chat_id: str, messages: List[Dict], title: Optional[str] = None, insights: Optional[List[Dict]] = None, pooled_data: Optional[Dict] = None, proactive_thought: Optional[str] = None, trace: Optional[Dict] = None, grade: Optional[Dict] = None):
        chat_data = self.load_chat(chat_id)
        if not chat_data:
            chat_data = {
                "id": chat_id,
                "created_at": datetime.now().isoformat(),
                "messages": [],
                "insights": [],
                "pooled_data": {},
                "proactive_thought": None,
                "trace": {},
                "grade": None
            }
        
        chat_data["messages"] = messages
        chat_data["updated_at"] = datetime.now().isoformat()
        if insights is not None:
            chat_data["insights"] = insights
        if pooled_data is not None:
            chat_data["pooled_data"] = pooled_data
        if proactive_thought is not None:
            chat_data["proactive_thought"] = proactive_thought
        if trace is not None:
            chat_data["trace"] = trace
        if grade is not None:
            chat_data["grade"] = grade
            
        if title:
            chat_data["title"] = title
        elif not chat_data.get("title") or chat_data["title"] == "New Chat":
            # Auto-title from first user message
            for msg in messages:
                if msg["role"] == "user":
                    if "versions" in msg:
                        content_data = msg["versions"][msg.get("active_version", 0)]
                        content = content_data["content"]
                    else:
                        content = msg.get("content", "")
                        
                    if isinstance(content, list):
                        content = content[0]
                    chat_data["title"] = content[:30] + ("..." if len(content) > 30 else "")
                    break
        
        self._save_chat(chat_data)

    def delete_chat(self, chat_id: str):
        path = os.path.join(self.chats_dir, f"{chat_id}.json")
        if os.path.exists(path):
            os.remove(path)

    def _save_chat(self, chat_data: Dict):
        path = os.path.join(self.chats_dir, f"{chat_data['id']}.json")
        with open(path, 'w') as f:
            json.dump(chat_data, f, indent=2)

    def export_as_markdown(self, chat_id: str) -> str:
        chat_data = self.load_chat(chat_id)
        if not chat_data:
            return ""
        
        md = f"# Research Brief: {chat_data.get('title', 'Untitled')}\n"
        md += f"Date: {chat_data.get('updated_at', 'Unknown')}\n\n"
        
        md += "## 👁️ Contextual Insights\n"
        for insight in chat_data.get("insights", []):
            md += f"- **{insight['term']}**: {insight['description']} (Relevance: {insight['relevance']}%)\n"
        
        md += "\n## 💬 Conversation History\n"
        for msg in chat_data.get("messages", []):
            role = "USER" if msg["role"] == "user" else "RAGGEDY"
            if "versions" in msg:
                content_data = msg["versions"][msg.get("active_version", 0)]
                content = content_data["content"]
                sources = content_data.get("sources", [])
            else:
                content = msg.get("content", "")
                sources = msg.get("sources", [])
                
            md += f"### {role}\n{content}\n\n"
            if sources:
                md += "**Sources:**\n"
                for src in sources:
                    md += f"- [{src['index']}] {src['filename']}\n"
                md += "\n"
        
        if chat_data.get("grade"):
            g = chat_data["grade"]
            md += f"## 🎯 Quality Grade\n"
            md += f"Total Score: {g.get('total_score')}/10\n"
            md += f"- Faithfulness: {g.get('faithfulness')}/10\n"
            md += f"- Grounding: {g.get('grounding')}/10\n"
            md += f"- Completeness: {g.get('completeness')}/10\n"
            md += f"\nCritique: {g.get('critique')}\n"
            
        return md
