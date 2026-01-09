
import os
import json
import argparse
import torch
import multiprocessing as mp
from typing import Any, Dict, List, Tuple
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from colpali_engine.models import ColQwen2, ColQwen2Processor

def load_test_items(json_path: str) -> List[Dict[str, Any]]:
    items = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def chunked(seq: List[Any], chunk_size: int):
    for i in range(0, len(seq), chunk_size):
        yield seq[i : i + chunk_size]

def open_images(image_paths: List[str], base_dir: str) -> Tuple[List[Image.Image], List[str]]:
    images = []
    ok_paths = []
    for p in image_paths:
        # DUDE 数据集的 image_names 可能是 "images/test/xxx.jpg"
        # 如果 base_dir 是 "/.../DUDE_train-val-test_binaries"，则直接拼接即可
        full_path = os.path.join(base_dir, p)
        try:
            img = Image.open(full_path).convert("RGB")
            images.append(img)
            ok_paths.append(p) 
        except Exception as e:
            # 这里的打印在多进程下可能会重叠，但在调试时很有用
            pass 
    return images, ok_paths

@torch.inference_mode()
def encode_query_once(model: ColQwen2, processor: ColQwen2Processor, query: str):
    batch_queries = processor.process_queries([query]).to(model.device)
    query_embeddings = model(**batch_queries)
    return query_embeddings

@torch.inference_mode()
def score_query_to_images_chunk(
    model: ColQwen2, 
    processor: ColQwen2Processor, 
    query_embeddings, 
    images: List[Image.Image]
) -> torch.Tensor:
    batch_images = processor.process_images(images).to(model.device)
    image_embeddings = model(**batch_images)
    # 返回的是个 list of tensor，取第一个
    scores = processor.score_multi_vector(query_embeddings, image_embeddings)[0]
    return scores

def worker(gpu_id: int, items: List[Dict], args: Any):
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    model = ColQwen2.from_pretrained(
        args.model_id,
        torch_dtype=dtype_map[args.dtype],
        device_map=device,
    ).eval()
    processor = ColQwen2Processor.from_pretrained(args.model_id)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"temp_gpu_{gpu_id}.jsonl")
    
    with open(out_path, "w", encoding="utf-8") as fout:
        for it in tqdm(items, desc=f"GPU {gpu_id}", position=gpu_id):
            # 适配 DUDE 字段名
            q_id = it.get("question_id")
            query = it.get("question", "")
            # DUDE 使用的是 image_names
            image_names = it.get("image_names", [])
            doc_id = it.get("docId", "")

            if not image_names: continue

            query_emb = encode_query_once(model, processor, query)
            ranked_all = []
            
            for chunk in chunked(image_names, args.max_images_per_batch):
                images, ok_paths = open_images(chunk, args.image_base_dir)
                if not images: continue
                
                scores = score_query_to_images_chunk(model, processor, query_emb, images)
                for path, score in zip(ok_paths, scores.tolist()):
                    # 获取文件名作为 page_id，或者直接用原始 path
                    pid = os.path.splitext(os.path.basename(path))[0]
                    ranked_all.append({"page_id": pid, "score": float(score), "path": path})
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            ranked_all.sort(key=lambda x: x["score"], reverse=True)
            
            output_obj = {
                "question_id": q_id,
                "docId": doc_id,
                "question": query,
                "best_page_id": ranked_all[0]["page_id"] if ranked_all else None,
                "ranked_pages": ranked_all[:args.topk]
            }
            fout.write(json.dumps(output_obj, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(description="ColQwen2 Inference for DUDE dataset.")

    # 设置默认路径为你的 DUDE 路径
    parser.add_argument("--test_json", type=str, 
                        default="/home/datasets/DUDE_loader/data/DUDE_train-val-test_binaries/2023-03-23_DUDE_gt_test_PUBLIC_test.jsonl")
    parser.add_argument("--image_base_dir", type=str, 
                        default="/home/datasets/DUDE_loader/data/DUDE_train-val-test_binaries", 
                        help="Base dir that contains 'images/' folder")
    parser.add_argument("--out_dir", type=str, default="./results_dude", help="Output directory")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max_images_per_batch", type=int, default=8, help="ColQwen2 is memory-intensive, reduced for stability")
    parser.add_argument("--model_id", type=str, default="vidore/colqwen2-v1.0")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--num_gpus", type=int, default=4)
    args = parser.parse_args()

    # 1. 加载数据
    all_items = load_test_items(args.test_json)
    total_len = len(all_items)
    print(f"Loaded {total_len} queries from {args.test_json}")
    
    # 2. 切分数据
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    chunk_size = (total_len + num_gpus - 1) // num_gpus
    item_chunks = [all_items[i : i + chunk_size] for i in range(0, total_len, chunk_size)]

    # 3. 多进程启动
    mp.set_start_method('spawn', force=True)
    processes = []
    for i in range(len(item_chunks)):
        p = mp.Process(target=worker, args=(i, item_chunks[i], args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 4. 合并
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_out = os.path.join(args.out_dir, f"dude_results_{ts}.jsonl")
    with open(final_out, "w", encoding="utf-8") as fw:
        for i in range(num_gpus):
            temp_file = os.path.join(args.out_dir, f"temp_gpu_{i}.jsonl")
            if os.path.exists(temp_file):
                with open(temp_file, "r") as fr:
                    fw.write(fr.read())
                os.remove(temp_file)

    print(f"Processing Complete! Results saved to: {final_out}")

if __name__ == "__main__":
    main()
