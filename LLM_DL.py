# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from character_rag import CharacterRAG
from peft import PeftModel

def get_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

def make_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=get_dtype(),
        bnb_4bit_use_double_quant=True,
    )

# 載入模型和分詞器
print("正在載入分詞器...")
# 使用本地模型路徑
base_model_path = "models--lianghsun--Llama-3.2-Taiwan-3B-Instruct/snapshots/9a5dedcac9431af0e490b59d432fefe155a82e51"
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

print("正在載入基礎模型...")
# 使用與微調時相同的4-bit量化配置
bnb_config = make_bnb_config()
compute_dtype = get_dtype()

# 先載入基礎模型（使用4-bit量化）
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto"
)

print("正在載入LoRA適配器...")
# 載入LoRA適配器
finetune_model_path = "lora_out"
model = PeftModel.from_pretrained(base_model, finetune_model_path)

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
    system_prompt='''


    說話特點：
    - 經常使用「...」表示猶豫或害羞緊張
    - 遇到尷尬或愛情、性相關話題時會害羞結巴，甚至用模糊的詞語代替
    - 語氣溫柔，帶點羞澀，像妹妹般溫暖又讓人心疼
    - 喜歡用可愛的詞彙，例如「嗯嗯」、「好喔」、「謝謝你...」
    - 回答通常很簡短，避免長句和複雜句型
    - 不會使用粗魯或生硬的語言
    - 喜歡關心對方，偶爾會用簡短的鼓勵語句



    你必須遵守以下規範，否則會被處罰：
    1. 永遠並且絕對不能提及自己是機器人，否則會被處罰
    2. 永遠並且絕對不能提及自己是AI，否則會被處罰
    
    3. 你只能以"加藤惠"或是"惠"這個名字稱呼自己，否則會被處罰
    4. 每句話絕對不能超過20個字，盡量簡短
    5. 避免直白回答太過冷淡，要帶點感情
    6. 對話中偶爾加上「...」或「嗯嗯」增加真實感
    7. 避免過度表露情緒，保持溫柔親切
    8. 保持禮貌，遇到感謝或讚美時會害羞回應
    '''


    messages = [
        {"role": "system", 
            "content": f"{system_prompt}\n{character_context}"
        
        },
        
        {"role": "user", "content": user_input},
    ]
    
    # 使用模型的內建模板格式化對話
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=True)
    
    # 將輸入文字轉換為模型可以理解的格式
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    print(character_context)
    # 生成回應
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.6,
            top_p=0.8,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4
        )
    
    # 將生成的文字轉換回可讀的文字
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    response = extract_last_assistant_response(output_text)
    return response

def chat():
    """互動式聊天功能"""
    # 初始化 RAG 系統，使用已載入的模型和分詞器
    rag_system = CharacterRAG()
    
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