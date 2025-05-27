from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os

class CharacterRAG:
    def __init__(self,model ,tokenizer):
        
        # 初始化 LlamaIndex 組件
        llm = HuggingFaceLLM(
            model=model,
            tokenizer=tokenizer    
        )
        
        embed_model = HuggingFaceEmbedding(
            model_name="moka-ai/m3e-base"
        )
        
        Settings.llm = llm 
        Settings.embed_model = embed_model
        
        self.index = None
        self.character_profile = None

    def load_character_data(self, data_dir):
        """載入角色設定資料"""
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"已創建資料目錄：{data_dir}")
            return False
            
        documents = SimpleDirectoryReader(data_dir).load_data()
        self.index = VectorStoreIndex.from_documents(documents)
        
        
        # 讀取角色設定文件
        character_file = os.path.join(data_dir, "character_profile.txt")
        if os.path.exists(character_file):
            with open(character_file, "r", encoding="utf-8") as f:
                self.character_profile = f.read()
        
        return True

    def query_character(self, query_text):
            
        query_engine = self.index.as_query_engine(
            similarity_top_k=3,  # 獲取最相關的3個片段
            response_mode="tree_summarize"#如果不要叫他生成只是回傳查詢結果就是"simple"，也不用設定Setting.llm，如果是要在這邊直接生成並結果就是"tree_summarize"或其他演算法  # 使用樹狀結構總結
        )
        response = query_engine.query(query_text)
        return str(response)

    def get_character_context(self, query_text):
        """獲取角色相關的上下文資訊"""
        if not self.character_profile:
            return "請先載入角色設定資料"
            
        # 獲取相關的角色資訊
        relevant_info = self.query_character(query_text)
        return relevant_info
        
        