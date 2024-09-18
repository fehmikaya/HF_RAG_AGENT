from typing import Any, Dict, List, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
import json
import requests

class CustomLlama3(LLM):
    url: str = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
    bearer_token: str
    max_new_tokens: int = 1024
    top_p: float = 0.7
    temperature: float = 0.1

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        json_body = {
          "inputs": prompt,
                  "parameters": {"max_new_tokens":self.max_new_tokens, "top_p":self.top_p, "temperature":self.temperature}
          }

        data = json.dumps(json_body)
        headers = {
          'Content-Type': 'application/json',
          'Authorization': f'Bearer {self.bearer_token}'
        }
        response = requests.request("POST", self.url, headers=headers, data=data)
        response.raise_for_status()
        try:
          result=json.loads(response.content.decode("utf-8"))[0]['generated_text']
          target_substring = '<|end_header_id|>'
          substring_length = len(target_substring)
          last_occurrence_index = result.rfind(target_substring)
          if last_occurrence_index != -1:
            return result[last_occurrence_index + substring_length:].strip()
        except:
          return response

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {"llmUrl": self.url}

    @property
    def _llm_type(self) -> str:
        return "CustomLlama3"