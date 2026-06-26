import json
import urllib.request
import urllib.error
from ml.application.ports.ai_provider import AIProvider


LMSTUDIO_HOST = "http://localhost:1234"


class LMStudioAdapter(AIProvider):
    def __init__(self, host: str = None):
        self.host = host or LMSTUDIO_HOST

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
            self._request("/v1/models")
            return True
        except Exception:
            return False

    def list_models(self) -> list:
        try:
            resp = self._request("/v1/models")
            data = json.loads(resp.read())
            return [m["id"] for m in data.get("data", [])]
        except Exception:
            return []

    def chat(self, messages: list, model: str, on_token=None) -> str:
        url = f"{self.host}/v1/chat/completions"
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
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    return full_text
                try:
                    obj = json.loads(data_str)
                    token = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if token:
                        full_text += token
                        if on_token:
                            on_token(token)
                except json.JSONDecodeError:
                    continue
        return full_text

    def start_server(self) -> bool:
        return False

    def pull_model(self, model: str) -> bool:
        return False

    def get_name(self) -> str:
        return "lmstudio"