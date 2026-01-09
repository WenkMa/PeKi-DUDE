import json
import os
import base64
from pathlib import Path
from vllm import LLM, SamplingParams
from PIL import Image

# --- 配置区 ---
JSON_PATH = "/home/PeKiRAG/predictions_with_questions.jsonl"
OUT_JSONL = "final_results_qwen_top_1.jsonl" # 建议先存为jsonl，方便断点续传

# 模型路径
MODEL_PATH = "/home/models/Qwen3-VL-2B-Instruct"

START_FROM = 0   
CHUNK_SIZE = 50  

# --- 辅助函数 ---
def get_first_ranked_page(item: dict):
    ranked = item.get("ranked_pages", [])
    if not ranked:
        return None, None
    # 拼接完整的图片路径
    return ranked[0].get("page_id"), os.path.join("/home/datasets/DUDE_loader/data/DUDE_train-val-test_binaries/images/test", ranked[0].get("page_id")+".jpg")

def count_done_lines(jsonl_path: str) -> int:
    p = Path(jsonl_path)
    if not p.exists():
        return 0
    done = 0
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    json.loads(line.rstrip().rstrip(',')) # 适配可能带逗号的情况
                    done += 1
                except:
                    break
    return done

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_data_generator(jsonl_path, start_idx):
    """生成器：适配输入 jsonl 的 question_id 字段"""
    base_prompt = "Answer the question using a single word or phrase. If you cannot answer the question, please answer \"Not Answerable\"."
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            current_idx = i + 1
            if current_idx < start_idx:
                continue
            
            line = line.strip()
            if not line: continue
            
            try:
                item = json.loads(line)
            except:
                continue
            
            pid, img_path = get_first_ranked_page(item)
            
            # 修改：这里适配你提供的输入字段 "question_id"
            q_id = item.get("question_id")
            
            if q_id and item.get("question") and img_path:
                if not os.path.exists(img_path):
                    print(f"Warning: Image not found {img_path}")
                    continue
                
                base64_str = encode_image_to_base64(img_path)
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"}
                            },
                            {"type": "text", "text": f"{item['question']}\n{base_prompt}"},
                        ],
                    }
                ]

                yield {
                    "raw_item": item,
                    "messages": messages,
                    "original_idx": current_idx
                }

def run_vllm_inference():
    done_lines = count_done_lines(OUT_JSONL)
    real_start_from = START_FROM + done_lines
    print(f"Done: {done_lines}, Resuming from line: {real_start_from}")

    print("Initializing vLLM...")
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        dtype="float16", 
        tensor_parallel_size=2,
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        }
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=64
    )

    Path(OUT_JSONL).parent.mkdir(parents=True, exist_ok=True)
    
    data_gen = load_data_generator(JSON_PATH, real_start_from)
    batch_buffer = []
    has_more = True

    while has_more:
        while len(batch_buffer) < CHUNK_SIZE:
            try:
                item = next(data_gen)
                batch_buffer.append(item)
            except StopIteration:
                has_more = False
                break
        
        if not batch_buffer:
            break
            
        print(f"Processing chunk of {len(batch_buffer)} items...")
        input_messages = [b["messages"] for b in batch_buffer]
        outputs = llm.chat(messages=input_messages, sampling_params=sampling_params, use_tqdm=True)
        
        with open(OUT_JSONL, "a", encoding="utf-8") as wf:
            for idx, output in enumerate(outputs):
                original_item = batch_buffer[idx]["raw_item"]
                generated_text = output.outputs[0].text.strip()
                
                # --- 核心修改：匹配要求的输出格式 ---
                record = {
                    "questionId": original_item["question_id"], # 映射回 questionId
                    "answer": generated_text,
                    "answer_confidence": 1, # 默认设为 1
                    "answer_abstain": False
                }
                
                # 为了符合你提供的带逗号的展示格式，这里每行存一个对象，并在末尾加逗号
                # 注意：标准的 jsonl 通常不带逗号，如果你需要最后合成一个大的 JSON Array，可以在后处理做。
                # 这里的代码按你给的示例，每个对象后面加了逗号。
                wf.write(json.dumps(record, ensure_ascii=False) + ", ")
        
        print(f"Saved {len(batch_buffer)} items. Last index: {batch_buffer[-1]['original_idx']}")
        batch_buffer = []

    print(f"Finished. Results saved in: {OUT_JSONL}")

if __name__ == "__main__":
    run_vllm_inference()