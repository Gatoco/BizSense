import subprocess
import json
import urllib.request
import urllib.error
import os
from ml.application.ports.ai_provider import AIProvider


OLLAMA_HOST = "http://localhost:11434"


class OllamaAdapter(AIProvider):
    def __init__(self, host: str = None):
        self.host = host or os.environ.get("OLLAMA_HOST", OLLAMA_HOST)

    def _request(self, path: str, data: dict = None, method: str = "GET"):
        url = f"{self.host}{path}"
        if data is not None:
            payload = json.dumps(data).encode('utf-8')
            req = urllib.request.Request(url, data=payload, method=method, headers={'Content-Type': 'application/json'})
        else:
            req = urllib.request.Request(url, method=method)
        return urllib.request.urlopen(req, timeout=5)

    def is_available(self) -> bool:
        try:
            self._request("/api/tags")
            return True
        except Exception:
            return False

    def list_models(self) -> list:
        try:
            resp = self._request("/api/tags")
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def chat(self, messages: list, model: str, on_token=None) -> str:
        url = f"{self.host}/api/chat"
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "stream": True
        }).encode('utf-8')

        req = urllib.request.Request(url, data=payload, method='POST', headers={'Content-Type': 'application/json'})
        resp = urllib.request.urlopen(req, timeout=120)

        full_text = ""
        buffer = ""
        for chunk in iter(lambda: resp.read(1), b''):
            buffer += chunk.decode('utf-8', errors='ignore')
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    token = obj.get("message", {}).get("content", "")
                    if token:
                        full_text += token
                        if on_token:
                            on_token(token)
                    if obj.get("done"):
                        return full_text
                except json.JSONDecodeError:
                    continue
        return full_text

    def start_server(self) -> bool:
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            import time
            for _ in range(10):
                time.sleep(0.5)
                if self.is_available():
                    return True
            return self.is_available()
        except Exception:
            return False

    def pull_model(self, model: str) -> bool:
        try:
            subprocess.run(
                ["ollama", "pull", model],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=600
            )
            return True
        except Exception:
            return False

    def get_name(self) -> str:
        return "ollama"