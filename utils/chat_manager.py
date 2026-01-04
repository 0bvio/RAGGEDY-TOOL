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
            existing_insights = chat_data.get("insights", [])
            # Merge and deduplicate by 'term' (case-insensitive)
            # Prioritize existing insights if they are marked as 'manual'
            insight_map = {i['term'].lower(): i for i in existing_insights if isinstance(i, dict) and 'term' in i}
            for new_i in insights:
                if isinstance(new_i, dict) and 'term' in new_i:
                    term_lower = new_i['term'].lower()
                    if term_lower in insight_map and insight_map[term_lower].get('manual'):
                        # Keep the manual one, maybe update relevance if it's higher? 
                        # For now, stay strict: manual means manual.
                        continue
                    insight_map[term_lower] = new_i
            chat_data["insights"] = list(insight_map.values())
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

    def update_insight(self, chat_id: str, old_term: str, new_data: Dict):
        """Updates or renames an insight. Marks it as manual to prevent overwriting."""
        chat_data = self.load_chat(chat_id)
        if not chat_data or "insights" not in chat_data:
            return
        
        new_data["manual"] = True
        insights = chat_data["insights"]
        
        # If the term changed, we might need to remove the old one and add the new one
        # to ensure no duplicates if we use a simple loop
        updated = False
        for i, insight in enumerate(insights):
            if insight.get("term", "").lower() == old_term.lower():
                insights[i] = new_data
                updated = True
                break
        
        if not updated:
            insights.append(new_data)
        
        chat_data["insights"] = insights
        self._save_chat(chat_data)

    def delete_insight(self, chat_id: str, term: str):
        """Removes an insight by term."""
        chat_data = self.load_chat(chat_id)
        if not chat_data or "insights" not in chat_data:
            return
        
        chat_data["insights"] = [i for i in chat_data["insights"] if i.get("term", "").lower() != term.lower()]
        self._save_chat(chat_data)

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
            if isinstance(insight, dict) and "term" in insight:
                term = insight.get("term", "Unknown")
                desc = insight.get("description", "No description")
                rel = insight.get("relevance", "N/A")
                md += f"- **{term}**: {desc} (Relevance: {rel}%)\n"
            else:
                # Fallback for unexpected format
                md += f"- {str(insight)}\n"
        
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
