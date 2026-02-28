import base64
import json
import re
import urllib.request
from typing import Dict, Any

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_ID = "qwen2.5-vl-7b-instruct"

class QwenVLMInterface:
    def __init__(self, base_url: str = LM_STUDIO_URL, model_id: str = MODEL_ID):
        self.base_url = base_url
        self.model_id = model_id
        self._verify_server()

    def _verify_server(self):
        try:
            req = urllib.request.Request("http://localhost:1234/v1/models")
            resp = urllib.request.urlopen(req, timeout=10)
            models = json.loads(resp.read())
            names = [m["id"] for m in models["data"]]
            if self.model_id not in names:
                raise RuntimeError(
                    f"Model '{self.model_id}' not found. Available: {names}"
                )
            print(f"Connected to LM Studio — model: {self.model_id}")
        except Exception as e:
            raise RuntimeError(f"Cannot reach LM Studio at localhost:1234: {e}")

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def predict_force(self, image_path: str, object_name: str) -> Dict[str, Any]:
        try:
            b64 = self._encode_image(image_path)

            prompt = f"""You are a robot gripper controller. Estimate the required grip force for this object.

TASK: Pick up and lift this object 30cm vertically using a parallel-jaw gripper.

Object name: {object_name}

Respond with ONLY a JSON object (no markdown, no extra text):
{{
  "object_type": "what you observe",
  "material": "your material estimate",
  "estimated_mass_grams": number,
  "fragility": "very_fragile/fragile/normal/durable/very_durable",
  "required_grip_force_newtons": number,
  "confidence": "low/medium/high",
  "reasoning": "one sentence explanation"
}}

IMPORTANT:
- required_grip_force_newtons must be a NUMBER
- Consider object weight, material friction, and fragility
- Typical forces: fragile 2-5N, normal 5-15N, heavy tools 15-40N"""

            payload = {
                "model": self.model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "max_tokens": 300,
                "temperature": 0.1,
            }

            req = urllib.request.Request(
                self.base_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=120)
            result = json.loads(resp.read())

            response_text = result["choices"][0]["message"]["content"].strip()
            parsed = self._parse_json_response(response_text)

            return {
                "success": True,
                "prediction": parsed,
                "raw_response": response_text,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "raw_response": None,
            }

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        # Strip markdown fences if present
        cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try extracting the first JSON object
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return self._regex_parse(text)

    def _regex_parse(self, text: str) -> Dict[str, Any]:
        patterns = {
            "object_type": r'"object_type":\s*"([^"]*)"',
            "material": r'"material":\s*"([^"]*)"',
            "estimated_mass_grams": r'"estimated_mass_grams":\s*(\d+\.?\d*)',
            "fragility": r'"fragility":\s*"([^"]*)"',
            "required_grip_force_newtons": r'"required_grip_force_newtons":\s*(\d+\.?\d*)',
            "confidence": r'"confidence":\s*"([^"]*)"',
            "reasoning": r'"reasoning":\s*"([^"]*)"',
        }

        result = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                val = match.group(1)
                if key in ("estimated_mass_grams", "required_grip_force_newtons"):
                    try:
                        result[key] = float(val)
                    except ValueError:
                        result[key] = None
                else:
                    result[key] = val
            else:
                result[key] = None
        return result


def test_interface():
    vlm = QwenVLMInterface()
    result = vlm.predict_force(
        "/Users/sarthak215s/VLM-Force-Prediction/images/001_angle1.jpg",
        "Empty plastic water bottle",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    test_interface()
