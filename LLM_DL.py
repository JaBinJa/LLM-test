# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from character_rag import CharacterRAG

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

def generate_response(user_input, character_context, max_length=2048):
    # 使用模型的內建對話模板
    messages = [
        {"role": "system", "content": character_context},
        {"role": "user", "content": user_input}
    ]
    
    # 使用模型的內建模板格式化對話
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # 將輸入文字轉換為模型可以理解的格式
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    # 生成回應
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )
    
    # 將生成的文字轉換回可讀的文字
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    response = extract_last_assistant_response(output_text)
    return response

def chat():
    """互動式聊天功能"""
    # 初始化 RAG 系統，使用已載入的模型和分詞器
    rag_system = CharacterRAG(model ,tokenizer)
    
    # 設定角色資料目錄
    character_data_dir = "character_data"
    
    # 載入角色資料
    if rag_system.load_character_data(character_data_dir):
        print("成功載入角色設定資料")
    else:
        print("請在 character_data 目錄中放入角色設定文件")
        return
    
    print("輸入 'quit' 或 'exit' 結束對話")
    
    
    
    while True:
        # 獲取使用者輸入
        user_input = input("\n您: ")
        
        # 檢查是否要結束對話
        if user_input.lower() in ['quit', 'exit']:
            print("再見！")
            break
        
        # 獲取角色上下文
        character_context = rag_system.get_character_context(user_input)
        
        # 生成回應
        print("正在思考...")
        response = generate_response(user_input, character_context)
        print("\n機器人:", response)
        
    

if __name__ == "__main__":
    chat()