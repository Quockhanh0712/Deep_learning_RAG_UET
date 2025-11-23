# llm_wrapper.py
import logging
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- Logging setup ---
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

        self.model_name = cfg.get("llm", {}).get("model_name", "BAAI/bge-large-en-v1.5")
        self.temperature = cfg.get("llm", {}).get("temperature", 0.0)
        self.max_tokens = cfg.get("llm", {}).get("max_tokens", 512)
        self.mode = cfg.get("llm", {}).get("mode", "local")  # Chỉ local mode

        # Load local model using Hugging Face
        self.model, self.tokenizer = self.load_local_model(self.model_name)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1,  # CPU, nếu muốn dùng GPU đổi thành device=0
        )
        logging.info(f"LLMWrapper initialized with local model {self.model_name}")

    def load_local_model(self, model_name: str):
        """
        Load local Hugging Face model for text generation.
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            return model, tokenizer
        except Exception as e:
            logging.warning(f"Failed to load model {model_name}: {e}")
            return None, None

    def generate(self, prompt: str) -> str:
        """
        Generate text from a prompt using local model, with truncation to avoid token overflow.
        """
        if self.generator is None:
            return "[Local model generate not implemented]"

        # Tokenize prompt
        tokens = self.tokenizer.encode(prompt, return_tensors="pt")
        max_model_tokens = self.model.config.n_positions  # GPT2 max context length

        # Truncate if too long
        if tokens.size(1) > max_model_tokens - self.max_tokens:
            tokens = tokens[:, -(max_model_tokens - self.max_tokens):]
            prompt = self.tokenizer.decode(tokens[0], skip_special_tokens=True)

        output = self.generator(
            prompt,
            max_new_tokens=self.max_tokens,
            do_sample=True,
            temperature=self.temperature,
        )
        return output[0]["generated_text"]

    def chat(self, prompt_history: list) -> str:
        """
        Multi-turn chat. prompt_history: list of dicts {"role": "user/assistant", "content": "..."}
        """
        if self.generator is None:
            return "[Local model chat not implemented]"

        # Combine history into a single prompt
        combined_prompt = "\n".join([f"{turn['role']}: {turn['content']}" for turn in prompt_history])
        return self.generate(combined_prompt)
