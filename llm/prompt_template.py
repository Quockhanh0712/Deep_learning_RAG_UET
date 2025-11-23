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
            "You are a smart AI assistant. Use the information from the document to answer accurately."
        )
    if not context.strip():
        context = "No information available in the document."

    template = (
        "{instruction}\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
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
            "You are a smart AI assistant. Use the information from the document to answer accurately."
        )

    prompt_parts = [instruction, ""]
    if not context.strip():
        context = "No information available in the document."

    if context:
        prompt_parts.append("Information from the document:\n" + context)
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
