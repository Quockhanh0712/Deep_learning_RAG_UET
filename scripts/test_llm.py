from llm.llm_wrapper import LLMWrapper

llm = LLMWrapper()  # dùng config mặc định
prompt = "Viết một đoạn văn ngắn về trí tuệ nhân tạo."
output = llm.generate(prompt)
print("Output:", output)

# Test multi-turn chat
history = [
    {"role": "user", "content": "Chào bạn, bạn có khỏe không?"},
    {"role": "assistant", "content": "Tôi khỏe, cảm ơn bạn! Bạn thế nào?"}
]
history.append({"role": "user", "content": "Hãy kể một câu chuyện ngắn về robot."})
chat_output = llm.chat(history)
print("Chat Output:", chat_output)
