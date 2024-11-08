from decimal import InvalidContext
import torch
import sys
from torch._C import DeserializationStorageContext
from torchvision import transforms
import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tenseal as ts
from typing import Dict
import time



start_time = time.time()
##################
# Client Helpers #
##################
# Create the TenSEAL security context
def create_ctx():
    poly_modulus_degree = 32768  # Sử dụng giá trị lớn hơn để hỗ trợ mã hóa chính xác
    coeff_mod_bit_sizes = [60,40,40,40,40,40,40,40,40,60]  # Định nghĩa hệ số độ chính xác
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree,-1, coeff_mod_bit_sizes)
    ctx.global_scale = 2 ** 40  # Global scale cho các tính toán
    ctx.generate_galois_keys()  # Sinh khóa galois cho phép xoay vòng
    return ctx


# Sample and preprocess an image
def load_input(image_path):
    transform = transforms.Compose([
        transforms.Resize((36, 36)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.81918], std=[0.24993])
    ])
    img = Image.open(image_path).convert("L")
    img_tensor = transform(img).view(36,36).tolist()
    tensor_size = sys.getsizeof(pickle.dumps(img_tensor))  # pickle để đo kích thước
    print(f"Kích thước ảnh trước khi mã hóa: {tensor_size} bytes")
    return img_tensor, img

# Helper for encoding the image
def prepare_input(ctx, plain_input):
    enc_input, windows_nb = ts.im2col_encoding(ctx, plain_input, 4, 4, 4)
    assert windows_nb == 81
    
    enc_input_serialized = enc_input.serialize()
    enc_size = sys.getsizeof(enc_input_serialized)  # Đổi sang KB
    print(f"Kích thước ảnh sau khi mã hóa (CKKSVector): {enc_size} bytes")
    return enc_input

################
# Server Model #
################
def encrypted_custom_activation(enc_x):
    squared = enc_x.square()
    term1 = squared.mul(0.1524)
    term2 = enc_x.mul(0.5)
    return term1 + term2 + 0.409


class OptimizedCNN:
    def __init__(self, parameters: Dict[str, list]):
        self.conv1_weight = parameters["conv1.weight"]
        self.conv1_bias = parameters["conv1.bias"]
        self.fc1_weight = parameters["fc1.weight"]
        self.fc1_bias = parameters["fc1.bias"]
        self.fc2_weight = parameters["fc2.weight"]
        self.fc2_bias = parameters["fc2.bias"]
        self.windows_nb = 81
    
    def forward(self, enc_x: ts.CKKSVector) -> ts.CKKSVector:
        channels1 = []
        
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):  
            y = enc_x.conv2d_im2col(kernel, self.windows_nb)+ bias
            channels1.append(y)
        
        out = ts.CKKSVector.pack_vectors(channels1)
        out=encrypted_custom_activation(out)
        out = out.mm_(self.fc1_weight) + self.fc1_bias
        out=encrypted_custom_activation(out)
        out = out.mm_(self.fc2_weight) + self.fc2_bias
        return out

    @staticmethod
    def prepare_input(context: bytes, ckks_vector: bytes) -> ts.CKKSVector:
        try:
            ctx = ts.context_from(context)
            enc_x = ts.ckks_vector_from(ctx, ckks_vector)
        except:
            raise DeserializationStorageContext("Cannot deserialize context or CKKS vector")
        try:
            _ = ctx.galois_keys()
        except:
            raise InvalidContext("The context doesn't hold Galois keys")
        return enc_x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

##################
# Server helpers #
##################
def load_parameters(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as f:
            parameters = pickle.load(f)
        print(f"Model loaded from '{file_path}'")
    except OSError as ose:
        print("Error:", ose)
        raise ose
    return parameters

parameters = load_parameters(r"C:\Users\vanvu\Downloads\DoAn\sourcecode\NewFolder\ver5\OptimizedCNN.pickle")
model = OptimizedCNN(parameters)

################
# Client Query #
################
# CKKS context generation
context = create_ctx()

# Load and preprocess an image
image_path = r'C:\Users\vanvu\Downloads\DoAn\dataset\ABIDE\test\tc\50030_0.png'
image, orig = load_input(image_path)

# Encode image with CKKS
encrypted_image = prepare_input(context, image)
print("Encrypted image ", encrypted_image)

# Display original image
plt.figure()
plt.title("Original Image")
plt.imshow(np.asarray(orig), cmap='gray')
plt.show()

# Prepare context for the server by making it public
server_context = context.copy()
server_context.make_context_public()

# Serialize context and ciphertext
server_context = server_context.serialize()
encrypted_image = encrypted_image.serialize()

client_query = {
    "data": encrypted_image,
    "context": server_context,
}

####################
# Server inference #
####################
encrypted_query = model.prepare_input(client_query["context"], client_query["data"])
encrypted_result = model(encrypted_query).serialize()

server_response = {
    "data": encrypted_result
}

###########################
# Client process response #
###########################
# Decrypt the result on client side
result = ts.ckks_vector_from(context, encrypted_result).decrypt()
print(result)

result = ts.ckks_vector_from(context, encrypted_result).decrypt()
print(result)

# Sử dụng softmax để chuyển đổi đầu ra thành xác suất
probs = torch.softmax(torch.tensor(result), dim=0)

# Tìm lớp dự đoán có xác suất cao nhất
predicted_class = torch.argmax(probs).item()

print(f"Predicted class: {predicted_class}, Probabilities: {probs.tolist()}")
end_time = time.time()
total_time = end_time - start_time
print("Total execution time:", total_time, "seconds")