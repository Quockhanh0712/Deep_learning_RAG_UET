# prompt_template.py
import yaml
import os

def load_prompt_config(config_path="config/config.yaml"):
    """
    Load prompt configuration from config.yaml.
    """
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("prompt", {})

# Load once, dùng toàn bộ module
PROMPT_CONFIG = load_prompt_config()

def format_prompt(context: str, question: str, instruction: str = None) -> str:
    """
    Format a 1-turn QA prompt using context and question.
    If instruction is not provided, use default from config.yaml.
    """
    if instruction is None:
        instruction = PROMPT_CONFIG.get(
            "instruction",
            "Bạn là trợ lý AI thông minh. Trả lời câu hỏi dựa trên thông tin từ tài liệu."
        )
    template = (
        "{instruction}\n\n"
        "Thông tin từ tài liệu:\n{context}\n\n"
        "Câu hỏi:\n{question}\n\n"
        "Trả lời:"
    )
    return template.format(instruction=instruction, context=context, question=question)

def format_chat_prompt(history: list, context: str = "", instruction: str = None) -> str:
    """
    Format a multi-turn chat prompt.
    history: list of dicts [{"role": "user/assistant", "content": "..."}]
    context: optional retrieved context
    instruction: optional override instruction
    """
    if instruction is None:
        instruction = PROMPT_CONFIG.get(
            "instruction",
            "Bạn là trợ lý AI thông minh. Trả lời dựa trên thông tin từ tài liệu nếu có."
        )

    prompt_parts = [instruction, ""]

    if context:
        prompt_parts.append("Thông tin từ tài liệu:\n" + context)
        prompt_parts.append("")

    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        prompt_parts.append(f"{role.capitalize()}: {content}")

    prompt_parts.append("Assistant:")  # vị trí LLM trả lời
    return "\n".join(prompt_parts)

class PromptTemplate:
    @staticmethod
    def format(context: str, question: str, instruction: str = None) -> str:
        from llm.prompt_template import format_prompt
        return format_prompt(context, question, instruction)

    @staticmethod
    def format_chat(history: list, context: str = "", instruction: str = None) -> str:
        from llm.prompt_template import format_chat_prompt
        return format_chat_prompt(history, context, instruction)
