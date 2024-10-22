import os
import pickle
import torch  # Thư viện PyTorch
from anytree import Node, RenderTree

def build_tree(value, node_name="root"):
    if isinstance(value, dict):
        node = Node(f"{node_name} (dict, keys={len(value)})")
        for sub_key, sub_value in value.items():
            sub_key_type = type(sub_key).__name__
            child_node = build_tree(sub_value, f"{sub_key} ({sub_key_type})")
            child_node.parent = node
    elif isinstance(value, list):
        node = Node(f"{node_name} (list, len={len(value)})")
        if len(value) > 0:
            for i, item in enumerate(value):
                item_type = type(item).__name__
                child_node = build_tree(item, f"Item {i} ({item_type})")
                child_node.parent = node
        else:
            Node("Empty List", parent=node)
    elif isinstance(value, torch.Tensor):
        node = Node(f"{node_name} (torch.Tensor, size={tuple(value.size())})")
    else:
        node = Node(f"{node_name} ({type(value).__name__})")
    return node

def get_tree_structure(root):
    tree_lines = []
    for pre, fill, node in RenderTree(root):
        tree_lines.append(f"{pre}{node.name}")
    return "\n".join(tree_lines)

def analyze_dictionary(file_path, output_text_file="tree_structure.txt"):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        if not isinstance(data, dict):
            print(f"File không chứa dữ liệu kiểu dictionary, mà là kiểu {type(data).__name__}.")
            return

        root = build_tree(data, "root (dict)")
        tree_structure = get_tree_structure(root)

        with open(output_text_file, 'w', encoding='utf-8') as f_out:
            f_out.write(f"Cấu trúc của dictionary trong file '{file_path}':\n")
            f_out.write(tree_structure)
        
        print(f"Cấu trúc cây đã được ghi vào file '{output_text_file}'.")
    
    except Exception as e:
        print(f"Đã xảy ra lỗi khi mở file: {e}")

# Đường dẫn đến file cần phân tích
file_path1 = r'C:\Users\vanvu\Downloads\DoAn\sourcecode\NewFolder\ver3\grayscale_final_model_state.pickle'
file_path2 = r'C:\Users\vanvu\Downloads\DoAn\sourcecode\NewFolder\ver3\ConvMNIST-0.1.pickle'
file_path3 = r'C:\Users\vanvu\Downloads\DoAn\sourcecode\NewFolder\ver3\converted_model_structure.pickle'
file_path4 = r'C:\Users\vanvu\Downloads\DoAn\sourcecode\NewFolder\ver5\test.pickle'
# Gọi hàm phân tích và xây dựng cây cấu trúc, ghi vào file text
#analyze_dictionary(file_path1, output_text_file="grayscale_final_model_structure.txt")
#analyze_dictionary(file_path3, output_text_file="converted_model_structure.txt")
analyze_dictionary(file_path4, output_text_file=r"C:\Users\vanvu\Downloads\DoAn\sourcecode\NewFolder\ver5\test.txt") 
