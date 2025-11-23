from pipeline.logger import log_query
from llm.prompt_template import format_prompt

context = "This is a test context from chunks."
question = "Who is the main protagonist?"
instruction = "Answer based on the context below."

prompt = format_prompt(context=context, question=question, instruction=instruction)
print("=== Prompt for LLM ===")
print(prompt)
print("=====================")
