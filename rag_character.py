from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import HuggingFaceLLM
from llama_index.embeddings import HuggingFaceEmbedding
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

class CharacterRAG:
    def __init__(self, model_name="lianghsun/Llama-3.2-Taiwan-3B-Instruct"):
        # 初始化模型和分詞器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        
        # 設定裝置
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        
        # 初始化 LlamaIndex 組件
        self.llm = HuggingFaceLLM(
            model=self.model,
            tokenizer=self.tokenizer,
            context_window=2048,
            max_new_tokens=512,
            generate_kwargs={"temperature": 0.7, "do_sample": True}
        )
        
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embed_model
        )
        
        self.index = None

    def load_character_data(self, data_dir):
        """載入角色設定資料"""
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"已創建資料目錄：{data_dir}")
            return False
            
        documents = SimpleDirectoryReader(data_dir).load_data()
        self.index = VectorStoreIndex.from_documents(
            documents,
            service_context=self.service_context
        )
        return True

    def query_character(self, query_text):
        """查詢角色相關資訊"""
        if self.index is None:
            return "請先載入角色設定資料"
            
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query_text)
        return str(response)

    def generate_response(self, user_input, character_context):
        """生成回應"""
        # 結合角色設定和使用者輸入
        prompt = f"角色設定：{character_context}\n\n使用者：{user_input}\n\n請根據角色設定回應："
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=2048,
                num_return_sequences=1,
                temperature=0.9,
                top_p=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

def main():
    # 初始化 RAG 系統
    rag_system = CharacterRAG()
    
    # 設定角色資料目錄
    character_data_dir = "character_data"
    
    # 載入角色資料
    if rag_system.load_character_data(character_data_dir):
        print("成功載入角色設定資料")
    else:
        print("請在 character_data 目錄中放入角色設定文件")
        return
    
    print("歡迎使用角色 RAG 系統！")
    print("輸入 'quit' 或 'exit' 結束對話")
    
    while True:
        user_input = input("\n您: ")
        
        if user_input.lower() in ['quit', 'exit']:
            print("再見！")
            break
        
        # 查詢角色相關資訊
        character_context = rag_system.query_character(user_input)
        
        # 生成回應
        print("正在思考...")
        response = rag_system.generate_response(user_input, character_context)
        print("\n角色:", response)

if __name__ == "__main__":
    main() 