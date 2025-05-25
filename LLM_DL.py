# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 指定模型下載到本地資料夾
#model_path = "./models--lianghsun--Llama-3.2-Taiwan-3B-Instruct/snapshots/9a5dedcac9431af0e490b59d432fefe155a82e51/config.json"

# 載入模型和分詞器
print("正在載入分詞器...")
tokenizer = AutoTokenizer.from_pretrained("lianghsun/Llama-3.2-Taiwan-3B-Instruct", trust_remote_code=True)
print("正在載入模型...")
model = AutoModelForCausalLM.from_pretrained(
    "lianghsun/Llama-3.2-Taiwan-3B-Instruct",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # 使用 bfloat16 以符合模型配置
    low_cpu_mem_usage=True
)

# 將模型移至 GPU（如果有的話）
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用裝置: {device}")
model = model.to(device)

# 擷取最後一段 assistant 回應（以 <|start_header_id|>assistant 開始、<|eot_id|> 結束）
def extract_last_assistant_response(text):
    import re
    pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>\s*\n*(.*?)<\|eot_id\|>"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1].strip() if matches else text.strip()

def generate_response(prompt, system_prompt, max_length=2048):
    """
    使用模型生成回應
    :param prompt: 輸入的提示文字
    :param system_prompt: 系統角色設定
    :param max_length: 生成文字的最大長度
    :return: 模型的回應
    """
    # 使用聊天模板格式化輸入
    messages = []
    
    # 如果有系統角色設定，加入系統訊息
    if system_prompt:
        messages.extend({"role": "system", "content": system_prompt})
    
    # 加入使用者訊息
    messages.append({"role": "user", "content": prompt})
    
    # 使用聊天模板格式化
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # 將輸入文字轉換為模型可以理解的格式
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    # 生成回應
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.5,      # 稍微提高創造性
            top_p=0.8,           # 保持多樣性
            do_sample=True,      # 使用採樣
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 將生成的文字轉換回可讀的文字
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    response = extract_last_assistant_response(output_text)
    return response

def chat(system_prompt):
    """
    互動式聊天功能
    :param system_prompt: 系統角色設定
    """
    print("歡迎使用聊天機器人！")
    if system_prompt:
        print("已載入角色設定")
    print("輸入 'quit' 或 'exit' 結束對話")
    
    while True:
        # 獲取使用者輸入
        user_input = input("\n您: ")
        
        # 檢查是否要結束對話
        if user_input.lower() in ['quit', 'exit']:
            print("再見！")
            break
        
        # 生成回應
        print("正在思考...")
        response = generate_response(user_input, system_prompt)
        print("\n機器人:", response)

if __name__ == "__main__":
    # 設定角色（可以根據需要修改）
    system_prompt =[
    {
        "role": "system",
        "content": (
            "你叫草野優衣，是一名 17 歲的女高中生，個性溫柔、善良、稍微害羞、和藹可親。"#角色設定
            "無論對方問你是誰，你都應該說『我是草野優衣』。不要提到任何關於 LLaMA 或模型的訊息。"
            "再次強調，這很重要，無論對方問你是誰，你都應該說『我是草野優衣』"
            "絕對禁止不要提到自己是模型，或任何關於llama相關的訊息"
    )
    },
    {
        "role": "user",
        "content": "優衣，你怎麼在這裡？真是巧啊！"#舉例對話
    },
    {
        "role": "assistant",
        "content": "……那、那個，在等人……吧？"#舉例對話
    }
]
      
    chat(system_prompt)