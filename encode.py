# import os
import pandas as pd
import json
import pickle
import argparse


# def get_queries(file_in):
#     query_list, query_indices = [], []
#     q_count = 0
#     for line in open(file_in, 'r', encoding="utf-8"):
#         item = json.loads(line.strip())
#         doc_page = item["page_indices"]
#         doc_layout = item["layout_indices"]
#         for qa in item["questions"]:
#             query_list.append(qa["Q"])
#             # tuple of question index, start/end indices of doc
#             query_indices.append((q_count, *doc_page, *doc_layout))
#             q_count += 1
#     return query_list, query_indices
def get_queries(file_in):
    df = pd.read_parquet(file_in)
    query_dedup = df[['query_id', 'query_text']].drop_duplicates(subset='query_id').sort_values('query_id').reset_index(drop=True)
    query_list = query_dedup['query_text'].tolist()
    query_ids = query_dedup['query_id'].tolist()
    return query_list, query_ids

# def get_pages(file_in, mode="vlm_text"):
#     q_list, q_indices = [], []
#     dataset_df = pd.read_parquet(file_in)
#     for row_index, row in dataset_df.iterrows():
#         q_list.append(row[mode])
#         q_indices.append(row_index)
#     return q_list, q_indices
def get_pages(file_in, mode="image_bytes"):
    df = pd.read_parquet(file_in)
    corpus_dedup = df[['corpus_id', mode]].drop_duplicates(subset='corpus_id').sort_values('corpus_id').reset_index(drop=True)
    corpus_list = corpus_dedup[mode].tolist()
    corpus_ids = corpus_dedup['corpus_id'].tolist()
    return corpus_list, corpus_ids

def get_layouts(file_in, mode="vlm_text"):
    q_list, q_indices = [], []
    dataset_df = pd.read_parquet(file_in)
    for row_index, row in dataset_df.iterrows():
        layout_type = row["type"]
        bbox = row["bbox"]
        page_id = row["page_id"]
        # page_size = row["page_size"]
        if mode == "image_binary":
            q_list.append(row["image_binary"])
        else:
            if layout_type in ["table", "image"]: q_list.append(row[mode])
            else: q_list.append(row["text"])
        q_indices.append((row_index, page_id, *bbox))
    return q_list, q_indices


def get_layouts_hybrid(file_in):
    q_img_list, q_img_indices = [], []
    q_txt_list, q_txt_indices = [], []
    dataset_df = pd.read_parquet(file_in)
    for row_index, row in dataset_df.iterrows():
        layout_type = row["type"]
        bbox = row["bbox"]
        page_id = row["page_id"]
        if layout_type in ["table", "image"]: 
            q_img_list.append(row["image_binary"])
            q_img_indices.append((row_index, page_id, *bbox))
        else:
            q_txt_list.append(row["text"])
            q_txt_indices.append((row_index, page_id, *bbox))
    return q_img_list, q_img_indices, q_txt_list, q_txt_indices


def get_retriever(model, bs):
    if model == "BGE":
        from text_wrapper import BGE
        bs = bs if bs != -1 else 256
        return BGE(bs=bs)
    elif model == "E5":
        from text_wrapper import E5
        bs = bs if bs != -1 else 256
        return E5(bs=bs)
    elif model == "GTE":
        from text_wrapper import GTE
        bs = bs if bs != -1 else 256
        return GTE(bs=bs)
    elif model == "Contriever":
        from text_wrapper import Contriever
        bs = bs if bs != -1 else 256
        return Contriever(bs=bs)
    elif model == "DPR":
        from text_wrapper import DPR
        bs = bs if bs != -1 else 256
        return DPR(bs=bs)
    elif model == "ColBERT":
        from text_wrapper import ColBERTReranker
        bs = bs if bs != -1 else 256
        return ColBERTReranker(bs=bs)

    elif model == "ColPali":
        from vision_wrapper import ColPaliRetriever
        bs = bs if bs != -1 else 10
        return ColPaliRetriever(bs=bs)

    elif model == "ColQwen":
        from vision_wrapper import ColQwen2Retriever
        bs = bs if bs != -1 else 8
        return ColQwen2Retriever(bs=bs)

    elif model == "DSE-docmatix":
        from vision_wrapper import DSE
        bs = bs if bs != -1 else 2
        return DSE(model_name="checkpoint/dse-phi3-docmatix-v2", bs=bs)

    elif model == "DSE-wikiss":
        from vision_wrapper import DSE
        bs = bs if bs != -1 else 2
        return DSE(model_name="checkpoint/dse-phi3-v1", bs=bs)

    else:
        raise ValueError("the model name is not correct!")


def initialize_args():
    '''
    Example: python encode.py BGE --mode vlm_text --encode query,page,layout
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Model name, e.g. BGE')
    parser.add_argument('--bs', type=int, default=-1)
    parser.add_argument('--encode_path', type=str, default='encode-500')
    parser.add_argument('--encode', type=str, default="query,page,layout")
    parser.add_argument('--mode', choices=['vlm_text', 'oct_text', 'image_binary', 'image_hybrid'], default='vlm_text')
    return parser.parse_args()



if __name__ == "__main__":
    # ["BGE", "E5", "GTE", "Contriever", "DPR", "ColBERT", "ColPali", "ColQwen", "DSE-docmatix", "DSE-wikiss"]
    args = initialize_args()
    model, mode, encode, encode_path, bs = args.model, args.mode, args.encode, args.encode_path, args.bs

    import os
    os.makedirs(encode_path, exist_ok=True)

    retriever = get_retriever(model, bs)

    DATA_FILE = "dataset/hotel_extract_500_1k2k.parquet"  

    if "query" in encode:
        query_list, query_ids = get_queries(DATA_FILE)
        encoded_query = retriever.embed_queries(query_list)
        print("number of queries encoded:", len(encoded_query))
        with open(f"{encode_path}/encoded_query_{model}.pkl", "wb") as f:
            pickle.dump((encoded_query, query_ids), f)
        print("query encoding done!")

    if "page" in encode:
        corpus_list, corpus_ids = get_pages(DATA_FILE, mode="image_bytes")
        encoded_corpus = retriever.embed_quotes(corpus_list)
        print("number of corpus images encoded:", len(encoded_corpus))
        with open(f"{encode_path}/encoded_page_{model}.pkl", "wb") as f:
            pickle.dump((encoded_corpus, corpus_ids), f)
        print("corpus encoding done!")

    # layout 部分删

    # 保存元信息
    df = pd.read_parquet(DATA_FILE)
    # 去掉 image_bytes 列（太大，评测时不需要）
    meta_cols = [c for c in df.columns if c != 'image_bytes']
    df_meta = df[meta_cols]
    meta_path = f"{encode_path}/gt_meta.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(df_meta, f)
    print(f"gt meta saved to {meta_path}, shape={df_meta.shape}")
