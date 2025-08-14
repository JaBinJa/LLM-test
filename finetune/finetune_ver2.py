#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QLoRA finetuning script for Llama-3.2-Taiwan-3B-Instruct on a single RTX 4070 (12GB).
- Works with CUDA 12.x, transformers >= 4.39, bitsandbytes >= 0.41, peft >= 0.10
- Loads base model in 4-bit and trains LoRA adapters only (QLoRA)
- Supports CSV (user,assistant) or JSONL ({"messages": [...]})
- Masks loss so only assistant replies are optimized

Run (Windows example):

python qlora_finetune_llama32_gf.py ^
  --model_name_or_path "E:\\Python\\TaiwanLLM2\\Llama-3.2-Taiwan-3B-Instruct" ^
  --train_file "E:\\Python\\TaiwanLLM2\\finetune\\123_formatted.csv" ^
  --output_dir "E:\\Python\\TaiwanLLM2\\lora_out" ^
  --num_train_epochs 3

Tip: To force HF cache to E: drive permanently (optional):
  setx HF_HOME "E:\\huggingface"
"""

import os
import sys
import argparse
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ---------------------------
# 輔助函數 (Helper Functions)
# ---------------------------
ASSISTANT_HEADER_TEXT = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def get_dtype():
    """
    獲取計算數據類型
    如果CUDA可用且支援bfloat16，則使用bfloat16，否則使用float16
    這樣可以優化記憶體使用和計算效率
    """
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def make_bnb_config():
    """
    創建BitsAndBytes 4位元量化配置
    使用4位元量化來大幅減少模型記憶體佔用
    - load_in_4bit: 啟用4位元量化
    - bnb_4bit_quant_type: 使用nf4量化類型（比fp4更穩定）
    - bnb_4bit_compute_dtype: 計算時使用的數據類型
    - bnb_4bit_use_double_quant: 啟用雙重量化進一步節省記憶體
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=get_dtype(),
        bnb_4bit_use_double_quant=True,
    )


def default_lora_config(r=8, alpha=16, dropout=0.05):
    """
    創建預設的LoRA配置
    LoRA (Low-Rank Adaptation) 是一種高效的微調方法，只訓練少量參數
    - r: LoRA的秩，控制適配器的複雜度
    - alpha: LoRA的縮放因子
    - dropout: 防止過擬合的dropout率
    - target_modules: 指定要應用LoRA的模組（注意力機制和MLP層）
    """
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力機制的投影層
            "gate_proj", "up_proj", "down_proj",      # MLP層的投影
        ],
    )


def build_messages_from_pair(user: str, assistant: str) -> List[Dict[str, str]]:
    """
    從用戶和助手對話對構建標準的消息格式
    將簡單的對話對轉換為符合聊天模板的格式
    """
    return [
        {"role": "user", "content": user.strip()},
        {"role": "assistant", "content": assistant.strip()},
    ]


def load_training_dataset(path: str) -> Dataset:
    """
    載入訓練數據集，支援多種格式：
    1) JSONL格式：包含"messages"欄位的JSON行文件
    2) CSV格式：包含user和assistant欄位的CSV文件
    3) CSV格式：恰好兩列，第一列為用戶，第二列為助手
    
    自動檢測文件格式並標準化為統一的messages格式
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl" or ext == ".json":
        ds = load_dataset("json", data_files=path, split="train")
    elif ext == ".csv":
        ds = load_dataset("csv", data_files=path, split="train")
    else:
        raise ValueError(f"Unsupported data file extension: {ext}")

    # 標準化為統一的messages欄位
    def _normalize(example):
        if "messages" in example and isinstance(example["messages"], list):
            return {"messages": example["messages"]}
        # CSV處理路徑
        cols = [c.lower() for c in ds.column_names]
        if "user" in cols and "assistant" in cols:
            u = example.get("user") or example.get("User")
            a = example.get("assistant") or example.get("Assistant")
            return {"messages": build_messages_from_pair(u, a)}
        elif len(ds.column_names) >= 2:
            c0, c1 = ds.column_names[:2]
            return {"messages": build_messages_from_pair(example[c0], example[c1])}
        else:
            raise ValueError("Unable to parse dataset: need 'messages' or user/assistant columns")

    ds = ds.map(_normalize, remove_columns=[c for c in ds.column_names if c != "messages"])
    return ds


# ---------------------------
# 分詞和標籤處理 (Tokenization & Labeling)
# ---------------------------
@dataclass
class ConvExample:
    """
    對話範例數據類別
    包含輸入ID、注意力遮罩和標籤
    """
    input_ids: List[int]      # 分詞後的輸入ID序列
    attention_mask: List[int]  # 注意力遮罩（1表示有效token，0表示padding）
    labels: List[int]          # 訓練標籤（-100表示不計算損失的token）


class ConversationFormatter:
    """
    對話格式化器
    負責將對話消息轉換為模型訓練所需的格式
    實現了損失遮罩，只對助手的回覆計算損失
    """
    def __init__(self, tokenizer, max_length: int = 1024):
        self.tok = tokenizer
        self.max_length = max_length
        # 預先分詞助手標題模式，用於損失遮罩
        self.assistant_hdr_ids = self.tok(ASSISTANT_HEADER_TEXT, add_special_tokens=False)["input_ids"]

    def _mask_to_assistant_only(self, input_ids: List[int]) -> List[int]:
        """
        創建標籤遮罩，使損失只應用於助手內容的token上
        找到最後一個助手標題，從該標題之後開始計算損失
        這樣可以確保模型只學習如何生成助手的回覆，而不是用戶的輸入
        """
        labels = [-100] * len(input_ids)  # 初始化所有標籤為-100（不計算損失）
        # 尋找助手標題ID的最後一次出現
        hdr = self.assistant_hdr_ids
        if not hdr:
            return labels
        # 從末尾開始滑動窗口搜索，提高魯棒性
        start = -1
        for i in range(len(input_ids) - len(hdr), -1, -1):
            if input_ids[i : i + len(hdr)] == hdr:
                start = i + len(hdr)
                break
        if start == -1:
            # 未找到助手標題 → 保持所有遮罩（安全預設）
            return labels
        for j in range(start, len(input_ids)):
            labels[j] = input_ids[j]  # 從助手標題後開始計算損失
        return labels

    def _format_one(self, messages: List[Dict[str, str]]) -> ConvExample:
        """
        格式化單個對話範例
        使用分詞器的聊天模板渲染對話樣本，然後進行分詞和標籤處理
        """
        # 使用分詞器的聊天模板渲染對話樣本
        rendered = self.tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        # 對渲染後的文本進行分詞
        enc = self.tok(
            rendered,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]
        labels = self._mask_to_assistant_only(input_ids)  # 應用損失遮罩
        assert len(input_ids) == len(attn) == len(labels)
        return ConvExample(input_ids, attn, labels)

    def __call__(self, examples: Dict[str, List[Any]]):
        """
        批量處理數據集範例
        對數據集中的每個對話進行格式化處理
        """
        # 批量映射數據集
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for msgs in examples["messages"]:
            ex = self._format_one(msgs)
            out["input_ids"].append(ex.input_ids)
            out["attention_mask"].append(ex.attention_mask)
            out["labels"].append(ex.labels)
        return out


@dataclass
class DataCollatorPad:
    """
    數據整理器，負責將不同長度的序列填充到相同長度
    這是訓練時批次處理的必要步驟
    """
    tokenizer: Any
    pad_token_id: int      # 填充token的ID
    label_pad_id: int = -100  # 標籤填充ID（-100表示不計算損失）

    def __call__(self, features: List[Dict[str, Any]]):
        """
        將批次中的特徵填充到相同長度
        確保批次中的所有序列具有相同的長度，便於並行計算
        """
        # 將輸入ID、注意力遮罩和標籤轉換為張量
        batch_input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        batch_attention = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        batch_labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        # 使用填充序列函數將所有序列填充到相同長度
        input_ids = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(batch_attention, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=self.label_pad_id)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ---------------------------
# 主程式 (Main Program)
# ---------------------------

def main():
    """
    主函數：執行完整的QLoRA微調流程
    包括參數解析、模型載入、數據處理、訓練配置和執行
    """
    # 解析命令行參數
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=r"E:\Python\TaiwanLLM2\finetune\models--lianghsun--Llama-3.2-Taiwan-3B-Instruct\snapshots\9a5dedcac9431af0e490b59d432fefe155a82e51")
    parser.add_argument("--train_file", type=str, default=r"E:\\Python\\TaiwanLLM2\\finetune\\123_formatted.csv",
                        help="jsonl or csv with fields 'messages' or user/assistant pairs")
    parser.add_argument("--output_dir", type=str, default=r"E:\\Python\\TaiwanLLM2\\lora_out")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 優化CUDA性能設置
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')

    # 載入分詞器
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        # 對於Llama風格的分詞器，使用eos作為pad token
        tokenizer.pad_token = tokenizer.eos_token

    # 載入數據集
    raw_ds = load_training_dataset(args.train_file)

    # 格式化數據集 → 分詞ID和標籤
    formatter = ConversationFormatter(tokenizer, max_length=args.max_length)
    tokenized = raw_ds.map(formatter, batched=True, remove_columns=["messages"])  # 返回包含列表的字典

    # Bitsandbytes 4位元配置 + 模型載入
    bnb_config = make_bnb_config()
    compute_dtype = get_dtype()

    # 載入預訓練模型，應用4位元量化
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # 準備k位元訓練（修復層正規化、梯度等）
    model = prepare_model_for_kbit_training(model)

    # 附加LoRA適配器
    lora_cfg = default_lora_config(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()  # 顯示可訓練參數數量

    # 數據整理器
    data_collator = DataCollatorPad(tokenizer=tokenizer, pad_token_id=tokenizer.pad_token_id)

    # 訓練參數配置
    training_args = TrainingArguments(
        output_dir=args.output_dir,                    # 輸出目錄
        num_train_epochs=args.num_train_epochs,       # 訓練輪數
        per_device_train_batch_size=args.per_device_train_batch_size,  # 每個設備的批次大小
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # 梯度累積步數
        learning_rate=args.learning_rate,             # 學習率
        lr_scheduler_type="cosine",                   # 學習率調度器類型
        warmup_ratio=0.03,                           # 預熱比例
        logging_steps=10,                             # 日誌記錄步數
        save_steps=200,                               # 保存檢查點步數
        save_total_limit=2,                           # 保存檢查點總數限制
        bf16=(compute_dtype == torch.bfloat16),      # 是否使用bfloat16
        fp16=(compute_dtype == torch.float16),        # 是否使用float16
        optim="paged_adamw_32bit",                   # 優化器（記憶體效率版本）
        gradient_checkpointing=True,                  # 梯度檢查點（節省記憶體）
        report_to=["none"],                           # 不報告到外部服務
        remove_unused_columns=False,                  # 不移除未使用的列
        ddp_find_unused_parameters=False,             # DDP相關設置
    )

    # 創建訓練器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 開始訓練
    trainer.train()

    # 保存LoRA適配器和分詞器（基礎模型保持4位元且凍結）
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Training done. LoRA adapters saved to:", args.output_dir)


if __name__ == "__main__":
    main()
