import pickle
import torch

def convert_tensor_to_list(tensor):
    """
    Chuyển đổi tensor thành danh sách.
    
    Args:
        tensor: Tensor cần chuyển đổi.

    Returns:
        Danh sách đã chuyển đổi.
    """
    return tensor.tolist()  # Chuyển đổi tensor thành danh sách

def push_up_single_length_items(value):
    """
    Đẩy các phần tử có độ dài 1 lên cấp trên.

    Args:
        value: Giá trị cần xử lý.

    Returns:
        Giá trị đã được xử lý.
    """
    if isinstance(value, list):
        if len(value) == 1:
            # Nếu độ dài là 1, đẩy phần tử lên
            return push_up_single_length_items(value[0])  # Tiếp tục đẩy lên nếu cần
        else:
            return [push_up_single_length_items(item) for item in value]  # Xử lý các phần tử khác
    return value  # Nếu không phải danh sách, trả lại giá trị gốc

def convert_model_state(input_file_path, output_file_path):
    """
    Chuyển đổi cấu trúc của model state từ file đầu vào sang định dạng mong muốn.

    Args:
        input_file_path: Đường dẫn đến file pickle cần chuyển đổi.
        output_file_path: Đường dẫn đến file pickle đầu ra.

    Returns:
        None
    """
    try:
        # Tải dữ liệu từ file đầu vào
        with open(input_file_path, 'rb') as f:
            model_state = pickle.load(f)

        # Kiểm tra kiểu dữ liệu
        if not isinstance(model_state, dict):
            print(f"File không chứa dữ liệu kiểu dictionary, mà là kiểu {type(model_state).__name__}.")
            return

        # Chuyển đổi tensor sang danh sách
        converted_structure = {}
        for key, value in model_state.items():
            if isinstance(value, torch.Tensor):
                converted_value = convert_tensor_to_list(value)
                
                # Đảo ngược chiều cho fc1.weight và fc2.weight
                if key == 'fc1.weight':
                    # Đảo chiều
                    converted_structure[key] = [list(i) for i in zip(*converted_value)]
                elif key == 'fc2.weight':
                    # Đảo chiều 
                    converted_structure[key] = [list(i) for i in zip(*converted_value)]
                else:
                    converted_structure[key] = push_up_single_length_items(converted_value)
            else:
                print(f"Cảnh báo: Giá trị cho khóa '{key}' không phải là tensor và sẽ không được chuyển đổi.")

        # Lưu cấu trúc đã chuyển đổi vào file đầu ra
        with open(output_file_path, 'wb') as f_out:
            pickle.dump(converted_structure, f_out)

        print(f"Dữ liệu đã được chuyển đổi và lưu vào file '{output_file_path}'.")

    except Exception as e:
        print(f"Đã xảy ra lỗi khi mở file: {e}")

# Đường dẫn đến file cần chuyển đổi
input_file_path = r'C:\Users\vanvu\Downloads\DoAn\sourcecode\NewFolder\ver5\final_model_state.pickle'
output_file_path = r'C:\Users\vanvu\Downloads\DoAn\sourcecode\NewFolder\ver5\OptimizedCNN.pickle'

# Gọi hàm chuyển đổi
convert_model_state(input_file_path, output_file_path)
