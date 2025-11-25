# llm_wrapper.py
import logging
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class LLMWrapper:
    def __init__(self, config_path="config/config.yaml"):
        # Load config
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # Đổi mặc định sang Qwen mới, không cần trust_remote_code
        self.model_name = cfg.get("llm", {}).get(
            "model_name", "Qwen/Qwen2.5-1.5B-Instruct"
        )
        self.temperature = float(cfg.get("llm", {}).get("temperature", 0.7))
        self.max_tokens = int(cfg.get("llm", {}).get("max_tokens", 256))

        device = "cuda" if torch.cuda.is_available() else "cpu"

        logging.info(f"Loading model {self.model_name} on {device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )

        self.device = device
        logging.info(f"LLMWrapper initialized with {self.model_name}")

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.model.config.max_position_embeddings - self.max_tokens
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text

    def chat(self, prompt_history: list) -> str:
        """
        prompt_history: list[{"role": "user"/"assistant", "content": "..."}]
        """
        # Format đúng kiểu instruction cho Qwen2.5
        chat_prompt = ""
        for turn in prompt_history:
            if turn["role"] == "user":
                chat_prompt += f"User: {turn['content']}\n"
            elif turn["role"] == "assistant":
                chat_prompt += f"Assistant: {turn['content']}\n"

        chat_prompt += "Assistant:"
        return self.generate(chat_prompt)
