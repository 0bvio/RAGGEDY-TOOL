import json
import requests
import subprocess
import time
import os
import re
import threading
from typing import List, Dict, Optional
from llm.manager import ModelManager
from utils.resource_monitor import ResourceMonitor
from utils.logger import raggedy_logger

class LLMClient:
    def __init__(self, base_url: str = "http://localhost:8080/v1"):
        self.base_url = base_url
        self.manager = ModelManager()
        self.monitor = ResourceMonitor()
        self.server_process = None
        self.current_model = None
        self.gpu_lock = threading.Lock()

    def is_available(self) -> bool:
        try:
            # Check if llama.cpp server is running at the base URL
            response = requests.get(f"{self.base_url}/models", timeout=2)
            if response.status_code == 200:
                return True
            response = requests.get(self.base_url, timeout=2)
            return response.status_code == 200
        except:
            return False

    def start_local_server(self, model_filename: str, n_ctx: int = 4096):
        if self.is_available():
            raggedy_logger.info("LLM Server already available.")
            return True
            
        # Resource Check
        est_size = self.monitor.estimate_model_size(model_filename)
        if not self.monitor.can_accommodate(est_size):
            raggedy_logger.error(f"Insufficient resources to load model {model_filename} (Est: {est_size:.1f} GB)")
            return False

        model_path = self.manager.get_model_path(model_filename)
        if not os.path.exists(model_path):
            raggedy_logger.error(f"Model path not found: {model_path}")
            return False

        try:
            # Check if llama-cpp-python is even installed
            import llama_cpp
        except ImportError:
            raggedy_logger.error("llama-cpp-python is not installed. Please follow instructions in userinstruction.md")
            return False

        # Try to find llama-cpp-python's server or just run a generic command
        # Ideally we use 'python3 -m llama_cpp.server --model path'
        cmd = [
            "python3", "-m", "llama_cpp.server", 
            "--model", model_path,
            "--host", "0.0.0.0",
            "--port", "8080",
            "--n_ctx", str(n_ctx)
        ]
        
        raggedy_logger.info(f"Starting LLM server with command: {' '.join(cmd)}")
        try:
            # Create a log file for this server run
            log_file_path = os.path.join("logs", f"llm_server_{int(time.time())}.log")
            log_file = open(log_file_path, "w")
            
            self.server_process = subprocess.Popen(
                cmd, 
                stdout=log_file, 
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True
            )
            
            # Wait for it to start
            for i in range(30):
                if self.is_available():
                    raggedy_logger.info(f"LLM server started successfully after {i*2} seconds.")
                    self.current_model = model_filename
                    return True
                if self.server_process.poll() is not None:
                    raggedy_logger.error("LLM server process terminated unexpectedly.")
                    # Try to read the last few lines of the log for immediate feedback
                    try:
                        with open(log_file_path, "r") as lf:
                            lines = lf.readlines()
                            last_lines = "".join(lines[-5:])
                            if "AttributeError" in last_lines and "sampler" in last_lines:
                                raggedy_logger.error("Detected known llama-cpp-python bug. Consider pinning to 0.3.1 in requirements.txt.")
                            if "out of memory" in last_lines.lower() or "vram" in last_lines.lower():
                                raggedy_logger.error("Possible VRAM issue. Try a smaller model or lower n_ctx.")
                            if "Py_RunMain" in last_lines or "libc.so" in last_lines:
                                raggedy_logger.error("Detected low-level crash (Segmentation Fault). This usually indicates model/library incompatibility or driver issues.")
                            if "unknown model architecture" in last_lines.lower():
                                arch_match = re.search(r"unknown model architecture: '(.+?)'", last_lines.lower())
                                arch_name = arch_match.group(1) if arch_match else "unknown"
                                raggedy_logger.error(f"Unsupported model architecture: '{arch_name}'. Your version of llama-cpp-python ({llama_cpp.__version__}) is too old to support this model.")
                            raggedy_logger.error(f"Last log lines:\n{last_lines}")
                    except:
                        pass
                    return False
                time.sleep(2)
            raggedy_logger.error("LLM server timed out while starting.")
            return False
        except Exception as e:
            raggedy_logger.error(f"Failed to start server: {e}")
            return False

    def stop_local_server(self):
        if self.server_process:
            raggedy_logger.info("Stopping local LLM server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            self.server_process = None
            self.current_model = None
            raggedy_logger.info("Local LLM server stopped.")

    def complete(self, prompt: str, grammar: Optional[str] = None, temperature: float = 0.1, top_p: float = 0.95, n_predict: int = 1024, stream: bool = False, timeout: int = 60):
        if not self.is_available():
            if stream:
                def mock_gen():
                    for word in self._mock_complete(prompt).split():
                        yield word + " "
                        time.sleep(0.05)
                return mock_gen()
            return self._mock_complete(prompt)
            
        payload = {
            "prompt": prompt,
            "max_tokens": n_predict,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            "stop": ["Question:", "User:", "Assistant:"]
        }
        if grammar:
            payload["grammar"] = grammar
            
        try:
            if stream:
                return self._stream_request(payload, timeout=timeout)
            
            with self.gpu_lock:
                response = requests.post(f"{self.base_url}/completions", json=payload, timeout=timeout)
                if response.status_code == 200:
                    return response.json()["choices"][0]["text"].strip()
                else:
                    raggedy_logger.error(f"LLM server returned {response.status_code}: {response.text}")
                    return f"Error: LLM server returned {response.status_code}"
        except requests.exceptions.Timeout:
            raggedy_logger.error("LLM server request timed out.")
            return "Error: LLM server request timed out."
        except Exception as e:
            raggedy_logger.error(f"Error connecting to LLM server: {e}")
            return f"Error connecting to LLM server: {e}"

    def _stream_request(self, payload: Dict, timeout: int = 60):
        def locked_stream():
            with self.gpu_lock:
                try:
                    response = requests.post(f"{self.base_url}/completions", json=payload, stream=True, timeout=timeout)
                    if response.status_code != 200:
                        yield f"Error: LLM server returned {response.status_code}"
                        return

                    for line in response.iter_lines():
                        if line:
                            line_str = line.decode('utf-8')
                            if line_str.startswith("data: "):
                                data_str = line_str[6:]
                                if data_str == "[DONE]":
                                    break
                                try:
                                    data = json.loads(data_str)
                                    token = data["choices"][0].get("text", "")
                                    yield token
                                except:
                                    continue
                except requests.exceptions.Timeout:
                    yield "Error: LLM server request timed out."
                except Exception as e:
                    yield f"Error connecting to LLM server: {e}"
        return locked_stream()

    def _mock_complete(self, prompt: str) -> str:
        # Mock responses for testing when no LLM is running
        if "Extract entities" in prompt:
            return json.dumps([
                {"source": "Sample Entity", "target": "Another Entity", "relation": "connected to", "confidence": 0.9}
            ])
        return "This is a mock response. To see real AI output, start a local llama.cpp server at " + self.base_url

    def extract_entities_relations(self, text: str) -> List[Dict]:
        if not self.is_available():
            raggedy_logger.warning("LLM server not available for entity/relation extraction. Returning empty list.")
            return []

        prompt = f"""Task: Extract entities and their semantic relationships from the text below.
Rules:
1. Return ONLY a valid JSON list of objects.
2. Each object must have keys: "source", "target", "relation", "confidence".
3. Confidence should be a float between 0 and 1.
4. If no relationships are found, return [].

Text: {text}

JSON:"""
        # Background tasks should have shorter timeouts to avoid hanging
        response = self.complete(prompt, timeout=20)
        try:
            # More robust JSON extraction
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return []
        except Exception as e:
            raggedy_logger.error(f"Failed to parse entities/relations JSON: {e}. Raw response: {response[:100]}...")
            return []

    def extract_insights(self, query: str, history: List[Dict]) -> List[Dict]:
        """Extracts key concepts, terms, and insights from the conversation."""
        if not self.is_available():
            return []
            
        history_text = ""
        for msg in history[-3:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            if "versions" in msg:
                content = msg["versions"][msg.get("active_version", 0)]["content"]
            else:
                content = msg.get("content", "")
            history_text += f"{role}: {content}\n"

        prompt = f"""Task: Extract key concepts, technical terms, or research ideas discussed in this conversation.
Provide a description for each and an estimate of its relevance to the user's latest query.

Conversation History:
{history_text}

Latest Query: {query}

Rules:
1. Return ONLY a valid JSON list of objects.
2. Each object must have keys: "term", "description", "relevance" (0-100).
3. If no insights, return [].

JSON:"""
        # Background tasks should have shorter timeouts
        response = self.complete(prompt, timeout=30)
        try:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return []
        except:
            return []

    def generate_answer(self, query: str, context_chunks: List[Dict], history: List[Dict] = None, system_prompt: str = None, stream: bool = False, **kwargs) -> str:
        # Sort context by index if available (from reranker)
        sources_section = ""
        if context_chunks:
            sources_section = "Sources:\n"
            for i, c in enumerate(context_chunks):
                sources_section += f"Source {i+1} (from {c['filename']}):\n{c['text']}\n\n"
        
        history_text = ""
        if history:
            history_text = "Previous conversation:\n"
            for msg in history[-5:]: # Only last 5 messages for context
                role = "User" if msg["role"] == "user" else "Assistant"
                
                # Handle versioned or legacy message format
                if "versions" in msg:
                    active_idx = msg.get("active_version", 0)
                    content_data = msg["versions"][active_idx]
                    content = content_data["content"]
                else:
                    content = msg.get("content", "")
                
                if isinstance(content, list):
                    # For history, use the current version (last one or active one)
                    content = content[-1] 
                history_text += f"{role}: {content}\n"
            history_text += "\n"

        if not system_prompt:
            system_prompt = """You are RAGGEDY, an offline knowledge intelligence assistant.
Your goal is to provide a grounded, factual answer based ONLY on the provided Sources.

Instructions:
1. Every claim you make MUST be supported by at least one Source.
2. Refer to sources using the format [Source X] at the end of every sentence that uses information from that source, e.g., "The project uses hybrid search [Source 1]."
3. Use multiple sources if necessary, e.g., [Source 1][Source 2].
4. Do NOT say "Source X says", just append the citation.
5. If the sources do not contain enough information, state that clearly.
6. Use the Chat History for context in follow-up questions."""

        prompt = f"""{system_prompt}

{history_text}
{sources_section}

Question: {query}
Answer:"""
        return self.complete(prompt, stream=stream, **kwargs)

    def grade_answer(self, query: str, answer: str, context_chunks: List[Dict]) -> Dict:
        """Evaluates the quality and faithfulness of a generated answer."""
        if not self.is_available() or not context_chunks:
            return {"score": "N/A", "reason": "LLM Offline or No Context"}

        context_text = "\n\n".join([f"Source {i+1}:\n{c['text']}" for i, c in enumerate(context_chunks)])

        prompt = f"""Task: Grade the following AI-generated answer based on its faithfulness to the provided Sources.

User Query: {query}

AI Answer: {answer}

Sources:
{context_text}

Evaluation Criteria:
1. Faithfulness (0-10): Does the answer only use information found in the sources? (Penalty for hallucinations)
2. Grounding (0-10): Are the citations [Source X] used correctly?
3. Completeness (0-10): Does it fully address the user query using available data?

Return ONLY a JSON object with keys: "faithfulness", "grounding", "completeness", "total_score" (average), and "critique" (brief explanation).

JSON:"""

        response = self.complete(prompt, timeout=30)
        try:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                # Ensure all keys exist
                for key in ["faithfulness", "grounding", "completeness", "total_score", "critique"]:
                    if key not in data: data[key] = 0 if key != "critique" else "Missing critique"
                return data
            return {"score": 0, "critique": "Failed to parse grading JSON"}
        except:
            return {"score": 0, "critique": "Error during grading"}
