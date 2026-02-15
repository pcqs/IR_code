import pickle
import json
from metric_eval import evaluate_page, evaluate_layout
from tqdm import tqdm
import argparse


def batch_dot_product(query_vec, passage_vecs):
    return passage_vecs @ query_vec


def load_pickle(file_in):
    # Load pickled files
    with open(file_in, "rb") as fq:
        return pickle.load(fq)


def initialize_args():
    '''
    Example: encode.py BGE --encode query,page,layout
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Model name, e.g. BGE')
    parser.add_argument('--encode_path', type=str, default='encode')
    parser.add_argument('--encode', type=str, default="query,page,layout")
    return parser.parse_args()


# if __name__ == "__main__":
#     # ["BGE", "E5", "GTE", "Contriever", "DPR", "ColBERT"]
#
#     args = initialize_args()
#     model, encode, encode_path = args.model, args.encode, args.encode_path
#
#     if model.startswith("Col"):
#         from metric_eval import colbert_score, pad_tok_len
#
#     encoded_query, query_indices = load_pickle(f"{encode_path}/encoded_query_{model}.pkl")
#
#     if "page" in encode:
#         encoded_page, page_indices = load_pickle(f"{encode_path}/encoded_page_{model}.pkl")
#     if "layout" in encode:
#         encoded_layout, layout_indices = load_pickle(f"{encode_path}/encoded_layout_{model}.pkl")
#
#     gt_list = []
#     for line in open("dataset/MMDocIR_annotations.jsonl", 'r', encoding="utf-8"):
#         item = json.loads(line.strip())
#         for qa in item["questions"]:
#             qa["domain"] = item["domain"]
#             gt_list.append(qa)
#
#     if len(gt_list) != len(query_indices):
#         raise ValueError("number of indexed question do not match ground-truth")
#
#     # To do this for every query in query_indices:
#     for (query_id, start_pid, end_pid, start_lid, end_lid) in tqdm(query_indices):
#         query_vec = encoded_query[query_id]
#
#         if "page" in encode:
#             page_vecs = encoded_page[start_pid:end_pid + 1]
#             if not model.startswith("Col"):
#                 scores_page = batch_dot_product(query_vec, page_vecs)
#             else:
#                 page_vecs_pad, masks_page = pad_tok_len(page_vecs)
#                 scores_page = colbert_score(query_vec, page_vecs_pad, masks_page)
#             gt_list[query_id]["scores_page"] = scores_page.tolist()
#
#         if "layout" in encode:
#             layout_vecs = encoded_layout[start_lid:end_lid + 1]
#             if not model.startswith("Col"):
#                 scores_layout = batch_dot_product(query_vec, layout_vecs)
#             else:
#                 layout_vecs_pad, masks_layout = pad_tok_len(layout_vecs)
#                 scores_layout = colbert_score(query_vec, layout_vecs_pad, masks_layout, use_gpu=True)
#             gt_list[query_id]["scores_layout"] = scores_layout.tolist()
#             gt_list[query_id]["layout_indices"] = layout_indices[start_lid:end_lid + 1]
#
#     if "page" in encode:
#         evaluate_page(gt_list, model_name=model, topk=1, metric="recall")
#         evaluate_page(gt_list, model_name=model, topk=3, metric="recall")
#         evaluate_page(gt_list, model_name=model, topk=5, metric="recall")
#
#     if "layout" in encode:
#         evaluate_layout(gt_list, model_name=model, topk=1, metric="recall")
#         evaluate_layout(gt_list, model_name=model, topk=5, metric="recall")
#         evaluate_layout(gt_list, model_name=model, topk=10, metric="recall")


if __name__ == "__main__":
    args = initialize_args()
    model, encode, encode_path = args.model, args.encode, args.encode_path

    if model.startswith("Col"):
        from metric_eval import colbert_score, pad_tok_len

    import numpy as np
    import pandas as pd
    import torch

    # ---- 1. 加载编码结果 + 元信息 ----
    encoded_query, query_ids = load_pickle(f"{encode_path}/encoded_query_{model}.pkl")
    encoded_page, corpus_ids = load_pickle(f"{encode_path}/encoded_page_{model}.pkl")
    df_meta = load_pickle(f"{encode_path}/gt_meta.pkl")

    # ---- 2. 构建 corpus_id → 下标映射 ----
    corpus_id2idx = {cid: i for i, cid in enumerate(corpus_ids)}

    # ---- 3. 预处理 corpus 嵌入 ----
    if model.startswith("Col"):
        page_vecs_pad, masks_page = pad_tok_len(encoded_page)
    else:
        corpus_matrix = np.array(encoded_page)

    # ---- 4. 每个 query 对全部 corpus 打分（只跑一遍）----
    query_scores = {}
    for i, qid in enumerate(tqdm(query_ids, desc="Scoring")):
        query_vec = encoded_query[i]
        if not model.startswith("Col"):
            scores = batch_dot_product(query_vec, corpus_matrix)
        else:
            scores = colbert_score(query_vec, page_vecs_pad, masks_page, use_gpu=True)
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        query_scores[qid] = np.array(scores)

    # ---- 5. 评测（整体 + 分桶，一次完成）----
    from metric_eval import evaluate_recall_bucketed
    evaluate_recall_bucketed(
        df_meta, query_scores, corpus_ids,
        model_name=model, topk_list=[1, 3, 5]
    )
