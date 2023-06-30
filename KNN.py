import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import fasttext
import pandas as pd

data_dir = "/home/kali/mpsd"

labels = []
texts = []

# Đọc dữ liệu từ thư mục "malicious_pure"
malicious_pure_dir = os.path.join(data_dir, "malicious_pure")
for file_name in os.listdir(malicious_pure_dir):
    file_path = os.path.join(malicious_pure_dir, file_name)
    if file_name.endswith(".ps1"):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read().rstrip('\n')  # Loại bỏ ký tự xuống dòng
            labels.append("malicious")
            texts.append(content)

# Đọc dữ liệu từ thư mục "powershell_benign_dataset"
powershell_benign_dir = os.path.join(data_dir, "powershell_benign_dataset")
for file_name in os.listdir(powershell_benign_dir):
    file_path = os.path.join(powershell_benign_dir, file_name)
    if file_name.endswith(".ps1"):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read().rstrip('\n')  # Loại bỏ ký tự xuống dòng
            labels.append("benign")
            texts.append(content)

# Đọc dữ liệu từ thư mục "mixed_malicious"
mixed_malicious_dir = os.path.join(data_dir, "mixed_malicious")
for file_name in os.listdir(mixed_malicious_dir):
    file_path = os.path.join(mixed_malicious_dir, file_name)
    if file_name.endswith(".ps1"):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read().rstrip('\n')  # Loại bỏ ký tự xuống dòng
            labels.append("malicious")
            texts.append(content)

# Sử dụng FastText để pre-train và chuyển đổi thành vector đầu vào
pretrain_data_path = "pretrain_data.txt"

with open(pretrain_data_path, "w", encoding="utf-8") as pretrain_data:
    for text in texts:
        pretrain_data.write(text + "\n")

model = fasttext.train_unsupervised(pretrain_data_path, model="skipgram")

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Chuyển đổi văn bản thành vector đầu vào
X_train_vectors = [model.get_sentence_vector(text.replace('\n', '')) for text in X_train]
X_test_vectors = [model.get_sentence_vector(text.replace('\n', '')) for text in X_test]


# Xây dựng và huấn luyện mô hình Decision Tree
dt_model = KNeighborsClassifier()
dt_model.fit(X_train_vectors, y_train)

# Dự đoán nhãn trên tập kiểm tra
y_pred = dt_model.predict(X_test_vectors)

# Tính toán các chỉ số đánh giá
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="malicious")
recall = recall_score(y_test, y_pred, pos_label="malicious")
f1 = f1_score(y_test, y_pred, pos_label="malicious")

# Tạo bảng kết quả
result_table = pd.DataFrame({'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                             'Score': [accuracy, precision, recall, f1]})
print(result_table)
