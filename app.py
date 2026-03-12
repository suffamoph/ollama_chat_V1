from flask import Flask, request, jsonify, send_from_directory
import httpx
import json
import traceback
import sys
import time
import re

app = Flask(__name__)
OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
INTERNAL_MODEL_LIST_TIMEOUT = 1.0

# 内部部署模型配置
INTERNAL_MODEL_API_URL = "http://10.65.2.107:8000/v1/chat/completions"
INTERNAL_MODEL_API_KEY = "4cad834b60ed11f925929a03998de5ea"
NO_THINKING_SYSTEM_PROMPT = "请直接输出最终答案，不要输出思考过程、推理链或<think>标签。"


def extract_thinking_from_text(text):
    """从文本中提取 <think>...</think> 思考内容。"""
    if not text:
        return ""

    match = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return ""

    return match.group(1).strip()


def remove_thinking_blocks(text):
    """移除文本中的 <think>...</think>，只保留最终回答。"""
    if not text:
        return ""
    cleaned = re.sub(r"<think>\s*.*?\s*</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def ensure_no_thinking_system_message(messages):
    """确保消息列表首条包含禁用思考输出的系统提示。"""
    if not messages:
        return [{'role': 'system', 'content': NO_THINKING_SYSTEM_PROMPT}]

    first_message = messages[0]
    if first_message.get('role') == 'system':
        content = str(first_message.get('content', ''))
        if '不要输出思考过程' in content or '<think>' in content or '推理链' in content:
            return messages

    return [{'role': 'system', 'content': NO_THINKING_SYSTEM_PROMPT}] + messages


def parse_enable_thinking(value, default=False):
    """解析前端传入的 enable_thinking 值。"""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    value_str = str(value).strip().lower()
    return value_str in ('1', 'true', 'yes', 'on')

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/api/models', methods=['GET'])
def get_models():
    """获取所有可用模型（Ollama本地 + 内部部署）"""
    try:
        models = []
        # 默认优先本地：有本地模型时不阻塞等待内部 API
        prefer_local = request.args.get('prefer_local', '1') != '0'
        
        # 获取 Ollama 模型列表
        try:
            with httpx.Client() as client:
                response = client.get(OLLAMA_TAGS_URL, timeout=10.0)
            
            if response.status_code == 200:
                result = response.json()
                ollama_models = result.get('models', [])
                
                # 处理 Ollama 模型
                model_capabilities = {
                    'qwen3-vl': True,
                    'qwen3:latest': False,
                    'qwen3:0.6b': False,
                    'qwen3:1.7b': False,
                    'qwen3:4b': False,
                    'qwen3:8b': False,
                    'qwen3:14b': False,
                    'qwen3:30b': False,
                    'qwen3:32b': False,
                    'qwen3:235b': False,
                    'qwen3.5': True,
                    'gemma3:latest': True,
                    'gemma3:4b': True,
                    'gemma3:12b': True,
                    'gemma3:27b': True,
                    'gemma3:270m': False,
                    'gemma3:1b': False,
                    'llava': True,
                    'llama3.2-vision': True,
                    'minicpm-v': True,
                    'moondream': True,
                    'bakllava': True,
                    'glm': True,
                    'deepseek-ocr': True,
                }
                
                for model in ollama_models:
                    name = model.get('name', '')
                    if 'embed' in name.lower():
                        continue
                    
                    is_multimodal = False
                    model_name_lower = name.lower()
                    for capability_model, multimodal in model_capabilities.items():
                        if (model_name_lower == capability_model or 
                            model_name_lower.startswith(capability_model) or 
                            capability_model in model_name_lower):
                            is_multimodal = multimodal
                            break
                    
                    capability_label = '🎨' if is_multimodal else '📝'
                    
                    models.append({
                        'name': name,
                        'type': 'ollama',
                        'capability': capability_label,
                        'is_multimodal': is_multimodal,
                    })
        except Exception as e:
            print(f"获取Ollama模型列表失败: {str(e)}", file=sys.stderr)

        if prefer_local and models:
            models.sort(key=lambda x: (x['type'] != 'ollama', x['name']))
            return jsonify({'models': models}), 200
        
        # 获取内部部署模型列表
        try:
            headers = {
                'Authorization': f'Bearer {INTERNAL_MODEL_API_KEY}',
                'Content-Type': 'application/json'
            }
            
            # 尝试调用内部模型API获取模型列表（假设有/models端点）
            with httpx.Client() as client:
                response = client.get(
                    'http://10.65.2.107:8000/v1/models',
                    headers=headers,
                    timeout=INTERNAL_MODEL_LIST_TIMEOUT
                )
            
            if response.status_code == 200:
                result = response.json()
                internal_models = result.get('data', [])
                
                for model in internal_models:
                    model_id = model.get('id', '')
                    # 判断是否为多模态模型
                    is_multimodal = 'vl' in model_id.lower() or 'vision' in model_id.lower()
                    capability_label = '🎨' if is_multimodal else '📝'
                    
                    models.append({
                        'name': model_id,
                        'type': 'internal',
                        'capability': capability_label,
                        'is_multimodal': is_multimodal,
                    })
        except Exception as e:
            print(f"获取内部模型列表失败: {str(e)}", file=sys.stderr)
            # 添加已知的内部模型
            models.append({
                'name': 'qwen3.5-27b-gptq-int4',
                'type': 'internal',
                'capability': '🎨',
                'is_multimodal': True,
            })
        
        # 按类型排序（本地Ollama在前，内部模型在后）
        models.sort(key=lambda x: (x['type'] != 'ollama', x['name']))
        
        return jsonify({'models': models}), 200
    
    except Exception as e:
        print(f"获取模型列表失败: {str(e)}", file=sys.stderr)
        return jsonify({'models': []}), 200

@app.route('/chat', methods=['POST'])
def chat():
    try:
        print("收到请求", file=sys.stderr)
        start_time = time.time()
        
        data = request.json
        model = data.get('model', 'gemma3:latest')
        model_source = data.get('model_source')
        enable_thinking = parse_enable_thinking(data.get('enable_thinking'), default=False)
        messages = data.get('messages', [])
        image_base64 = data.get('image_base64')
        
        print(f"模型: {model}, 来源: {model_source}, thinking: {enable_thinking}, 消息数: {len(messages)}, 有图片: {image_base64 is not None}", file=sys.stderr)
        
        # 判断是否为内部部署模型：优先使用前端传入的来源，避免同名模型误判
        if model_source == 'internal':
            is_internal_model = True
        elif model_source == 'ollama':
            is_internal_model = False
        else:
            # 向后兼容旧前端：
            # 1) 带冒号的模型通常是 Ollama 标签（如 qwen3.5:9b）
            # 2) 仅对明确的内部模型名做匹配
            model_lower = str(model).lower()
            if ':' in model_lower:
                is_internal_model = False
            else:
                internal_models = ['qwen3.5-27b-gptq-int4', 'qwen3-vl']
                is_internal_model = any(model_lower == m or model_lower.startswith(m) for m in internal_models)
        
        if is_internal_model:
            # 调用内部模型API（OpenAI兼容格式）
            return call_internal_model(model, messages, image_base64, start_time, enable_thinking)
        else:
            # 调用Ollama本地模型
            return call_ollama_model(model, messages, image_base64, start_time, enable_thinking)
    
    except Exception as e:
        print(f"❌ 错误: {str(e)}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500

def call_ollama_model(model, messages, image_base64, start_time, enable_thinking=False):
    """调用Ollama本地模型"""
    try:
        # 默认关闭思考输出，开启时不注入限制提示
        if not enable_thinking:
            messages = ensure_no_thinking_system_message(messages)

        # 构建消息 - 如果有图片，添加到消息对象中
        api_messages = []
        for i, msg in enumerate(messages):
            api_msg = {
                'role': msg.get('role'),
                'content': msg.get('content')
            }
            # 如果是最后一条用户消息且有图片，添加 images 字段
            if (i == len(messages) - 1 and 
                msg.get('role') == 'user' and 
                image_base64):
                api_msg['images'] = [image_base64]
                print(f"✓ 将图片添加到消息对象", file=sys.stderr)
            
            api_messages.append(api_msg)
        
        # 构建 REST API 请求
        payload = {
            'model': model,
            'messages': api_messages,
            'stream': False
        }
        
        print(f"向 Ollama REST API 发送请求...", file=sys.stderr)
        
        # 使用 httpx 调用 REST API
        with httpx.Client() as client:
            response = client.post(
                OLLAMA_API_URL,
                json=payload,
                timeout=300.0  # 5分钟超时
            )
        
        if response.status_code != 200:
            error_text = response.text
            print(f"Ollama API 错误 ({response.status_code}): {error_text}", file=sys.stderr)
            return jsonify({
                'error': f"Ollama API 错误: {error_text}",
                'status_code': response.status_code
            }), response.status_code
        
        result = response.json()
        end_time = time.time()
        response_time = end_time - start_time
        
        # 从 Ollama 响应中提取 token 信息和耗时信息
        prompt_tokens = result.get('prompt_eval_count', 0)
        completion_tokens = result.get('eval_count', 0)
        total_tokens = prompt_tokens + completion_tokens
        
        # Ollama 返回的耗时是纳秒，转换为毫秒
        ollama_eval_duration = result.get('eval_duration', 0) / 1_000_000  # 纳秒转毫秒
        
        print(f"✓ 成功收到Ollama响应: {total_tokens} tokens, {response_time:.2f}s", file=sys.stderr)

        raw_response_content = result['message']['content']
        thinking_content = extract_thinking_from_text(raw_response_content)
        response_content = remove_thinking_blocks(raw_response_content)
        thinking_enabled = bool(thinking_content)
        
        return jsonify({
            'response': response_content,
            'model': model,
            'model_source': 'ollama',
            'thinking_requested': enable_thinking,
            'response_time': round(response_time, 2),
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'eval_duration': round(ollama_eval_duration, 0),
            'thinking_enabled': thinking_enabled,
            'thinking_content': thinking_content
        })
    
    except Exception as e:
        print(f"❌ Ollama错误: {str(e)}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500

def call_internal_model(model, messages, image_base64, start_time, enable_thinking=False):
    """调用内部部署模型（OpenAI兼容API）"""
    try:
        # 默认关闭思考输出，开启时不注入限制提示
        if not enable_thinking:
            messages = ensure_no_thinking_system_message(messages)

        # 转换消息格式为OpenAI格式（支持多种内容类型）
        api_messages = []
        for i, msg in enumerate(messages):
            api_msg = {
                'role': msg.get('role'),
                'content': []
            }
            
            # 添加文本内容
            content_text = msg.get('content', '')
            if content_text:
                api_msg['content'].append({
                    'type': 'text',
                    'text': content_text
                })
            
            # 如果是最后一条用户消息且有图片，添加图片URL
            if (i == len(messages) - 1 and 
                msg.get('role') == 'user' and 
                image_base64):
                api_msg['content'].append({
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/jpeg;base64,{image_base64}'
                    }
                })
                print(f"✓ 将图片添加到内部模型消息", file=sys.stderr)
            
            # 如果content是列表且为空，转为字符串
            if not api_msg['content']:
                api_msg['content'] = content_text
            elif len(api_msg['content']) == 1 and api_msg['content'][0]['type'] == 'text':
                api_msg['content'] = api_msg['content'][0]['text']
            
            api_messages.append(api_msg)
        
        # 构建请求（关闭思考模式）
        # Qwen3.5 文档建议：
        # 1) OpenAI兼容(vLLM/SGLang): extra_body.chat_template_kwargs.enable_thinking=false
        # 2) Alibaba Cloud兼容: enable_thinking=false
        payload = {
            'model': model,
            'messages': api_messages,
            'temperature': 0.7,
            'top_p': 0.8,
            'enable_thinking': enable_thinking,
            'extra_body': {
                'top_k': 20,
                'chat_template_kwargs': {
                    'enable_thinking': enable_thinking
                }
            }
        }
        
        print(f"向内部模型API发送请求: {INTERNAL_MODEL_API_URL}", file=sys.stderr)
        
        # 调用内部模型API
        headers = {
            'Authorization': f'Bearer {INTERNAL_MODEL_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        with httpx.Client() as client:
            response = client.post(
                INTERNAL_MODEL_API_URL,
                json=payload,
                headers=headers,
                timeout=300.0  # 5分钟超时
            )

            # 兼容部分严格校验字段的服务：若因 extra_body 报错则回退重试
            if response.status_code == 400:
                fallback_payload = {
                    'model': model,
                    'messages': api_messages,
                    'temperature': 0.7,
                    'top_p': 0.8,
                    'enable_thinking': enable_thinking,
                }
                response = client.post(
                    INTERNAL_MODEL_API_URL,
                    json=fallback_payload,
                    headers=headers,
                    timeout=300.0
                )
        
        if response.status_code != 200:
            error_text = response.text
            print(f"内部模型API错误 ({response.status_code}): {error_text}", file=sys.stderr)
            return jsonify({
                'error': f"内部模型API错误: {error_text}",
                'status_code': response.status_code
            }), response.status_code
        
        result = response.json()
        end_time = time.time()
        response_time = end_time - start_time
        
        # 从内部模型API响应中提取 token 信息
        usage = result.get('usage', {})
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens)
        
        # 提取响应内容/思考内容
        response_content = ''
        thinking_content = ''
        choices = result.get('choices', [])
        if choices and len(choices) > 0:
            message = choices[0].get('message', {})
            raw_content = message.get('content', '')

            # content 可能是字符串或分段数组
            if isinstance(raw_content, list):
                text_parts = []
                for item in raw_content:
                    if isinstance(item, dict):
                        if item.get('type') == 'text' and item.get('text'):
                            text_parts.append(item.get('text'))
                    elif isinstance(item, str):
                        text_parts.append(item)
                response_content = "\n".join(text_parts).strip()
            else:
                response_content = raw_content or ''

            reasoning_content = message.get('reasoning_content') or message.get('thinking') or ''
            if isinstance(reasoning_content, list):
                reasoning_content = "\n".join(str(x) for x in reasoning_content)

            # 某些服务会把思考混在 content 里
            if not reasoning_content:
                reasoning_content = extract_thinking_from_text(response_content)

            thinking_content = (reasoning_content or '').strip()

            # 正文中移除思考块，避免前端正文出现“思考内容”
            response_content = remove_thinking_blocks(response_content)

        thinking_enabled = bool(thinking_content)
        
        print(f"✓ 成功收到内部模型响应: {total_tokens} tokens, {response_time:.2f}s", file=sys.stderr)
        
        return jsonify({
            'response': response_content,
            'model': model,
            'model_source': 'internal',
            'thinking_requested': enable_thinking,
            'response_time': round(response_time, 2),
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'thinking_enabled': thinking_enabled,
            'thinking_content': thinking_content
        })
    
    except Exception as e:
        print(f"❌ 内部模型错误: {str(e)}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500

if __name__ == '__main__':
    print("启动 Flask 服务器...", file=sys.stderr)
    # Use threaded mode so a slow /chat call won't block / or /api/models.
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True, use_reloader=False)
