import ollama

def chat_with_ollama():
    # 简单对话
    response = ollama.chat(
        model='gemma3:latest',
        messages=[
            {
                'role': 'user',
                'content': '你好，请介绍一下你自己'
            }
        ]
    )
    print(response['message']['content'])

def stream_chat():
    # 流式输出（像打字机一样逐字显示）
    stream = ollama.chat(
        model='gemma3:latest',
        messages=[{'role': 'user', 'content': '写一首短诗'}],
        stream=True
    )
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
    print()  # 换行

def multi_turn_chat():
    # 多轮对话
    messages = []
    print("开始对话（输入 quit 退出）")
    
    while True:
        user_input = input("\n你: ")
        if user_input.lower() == 'quit':
            break
            
        messages.append({'role': 'user', 'content': user_input})
        
        response = ollama.chat(
            model='gemma3:latest',
            messages=messages
        )
        
        assistant_msg = response['message']['content']
        messages.append({'role': 'assistant', 'content': assistant_msg})
        
        print(f"\nAI: {assistant_msg}")

if __name__ == "__main__":
    print("=== 简单对话 ===")
    chat_with_ollama()
    
    print("\n=== 流式输出 ===")
    stream_chat()
    
    print("\n=== 多轮对话 ===")
    multi_turn_chat()
