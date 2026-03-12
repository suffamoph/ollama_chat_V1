#!/usr/bin/env python3
"""
测试图像上传功能
"""
import httpx
import base64
import sys
from pathlib import Path

# 创建一个简单的测试图像（1x1 红色像素）
def create_test_image():
    """创建一个简单的 PNG 图像用于测试"""
    # 这是一个最小的 1x1 红色 PNG (base64编码)
    png_data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    )
    test_image_path = Path("test_image.png")
    test_image_path.write_bytes(png_data)
    return str(test_image_path)

def test_image_chat():
    """测试带图像的聊天请求"""
    # 创建测试图像
    image_path = create_test_image()
    
    # 读取图像并转换为 base64
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    print(f"✓ 已创建测试图像: {image_path}")
    print(f"✓ 图像大小: {len(image_bytes)} 字节")
    print(f"✓ Base64 长度: {len(image_base64)}")
    
    # 发送请求
    payload = {
        "model": "gemma3:latest",
        "messages": [
            {
                "role": "user",
                "content": "这是什么图像？请描述一下。"
            }
        ],
        "image_base64": image_base64
    }
    
    print("\n发送请求至 http://127.0.0.1:5000/chat ...")
    print(f"模型: {payload['model']}")
    print(f"消息: {payload['messages'][0]['content']}")
    print(f"有图像: 是 (大小: {len(image_base64)} 字符)")
    
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                "http://127.0.0.1:5000/chat",
                json=payload
            )
        
        print(f"\n响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ 成功！")
            print(f"响应内容: {result.get('response', '没有内容')[:200]}...")
        else:
            print(f"✗ 错误！")
            print(f"错误信息: {response.text}")
            
    except Exception as e:
        print(f"✗ 请求失败: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    test_image_chat()
