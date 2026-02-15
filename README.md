dataset：https://huggingface.co/datasets/AAAzoblue/hotel_extract_500_1k2k

command:
python encode.py ColQwen --encode page,query --bs 8 
python search.py ColQwen --encode page --encode_path encode-500
