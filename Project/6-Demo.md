## 6. Triển khai ứng dụng Web:

Sau khi hoàn tất quá trình huấn luyện mô hình phân tích cảm xúc tiếng Việt, nhóm tiến hành triển khai mô hình lên ứng dụng web nhằm cho phép người dùng tương tác trực tiếp với hệ thống.  
Phần giao diện người dùng (UI) được xây dựng bằng thư viện **Gradio**, và toàn bộ ứng dụng được triển khai trên nền tảng **Hugging Face Spaces**, giúp người dùng có thể truy cập và sử dụng trực tuyến mà không cần cài đặt phức tạp.

🔗 **Link ứng dụng web**:  
[Vietnamese Student Sentiment Analysis - a Hugging Face Space by oriontk24](https://huggingface.co/spaces/oriontk24/Vietnamese-Sentiment-Analysis)
![Demo](img/6-Hinh1.png)
_Hình 6.1: Giao diện ứng dụng web phân tích cảm xúc tiếng Việt triển khai bằng Gradio trên Hugging Face Spaces._
## 6.1. Kiến trúc dự án:
```
project/  
├── app.py                     # Gradio UI + xử lý inference  
├── phobert_production_bundle.zip  # Model + tokenizer đã đóng gói  
├── requirements.txt          # Danh sách thư viện cần thiết  
├── sample_comments.csv       # Dữ liệu mẫu để test batch inference  
├── test_phobert.py           # Script test model  
└── notebook.ipynb            # File huấn luyện model
```
## 6.2. Quy trình xử lý và dự đoán:

Luồng hoạt động của hệ thống được mô tả như sau:
### Single Inference (Dự đoán đơn lẻ):

1. Người dùng nhập một đoạn văn bản tiếng Việt vào giao diện
2. Văn bản được xử lý và tokenize bằng **PhoBERT tokenizer**
3. Dữ liệu được đưa vào mô hình đã huấn luyện
4. Hệ thống trả về kết quả dự đoán dưới dạng xác suất cho từng lớp:
    - Positive 😊
    - Neutral 😐
    - Negative 😡
5. Kết quả được hiển thị trực quan dưới dạng **Bar Chart**
### Batch Inference (Dự đoán hàng loạt):

1. Người dùng upload file CSV/Excel chứa nhiều bình luận
2. Hệ thống tự động:
    - Nhận diện cột chứa văn bản
    - Tiền xử lý dữ liệu
3. Thực hiện dự đoán hàng loạt
4. Trả về:
    - File kết quả đã gán nhãn
    - Biểu đồ phân bố cảm xúc (**Pie Chart**)
## 6.3. Quy trình triển khai (Deployment Pipeline):

1. **Model Packaging**:  
    Mô hình đã huấn luyện (`keras_model`) và tokenizer được đóng gói thành file `.zip` nhằm tối ưu dung lượng lưu trữ và dễ dàng deploy.
2. **Model Loading**:  
    Khi ứng dụng khởi chạy, hệ thống tự động giải nén file vào thư mục `model/phobert_bundle` và load model bằng `tf.keras`.
3. **UI Integration**:  
    Gradio được sử dụng để xây dựng giao diện tương tác trực quan, hỗ trợ cả dự đoán đơn và batch.
4. **Cloud Deployment**:  
    Source code (`app.py`), model và `requirements.txt` được upload lên Hugging Face Spaces để chạy online.

## 6.4. Hướng dẫn chạy local:

Bạn có thể clone và chạy project trên máy cá nhân theo các bước sau:
```
# 1. Clone source code  
git clone https://huggingface.co/spaces/oriontk24/Vietnamese-Sentiment-Analysis  
cd Vietnamese-Sentiment-Analysis  
  
# 2. Cài đặt thư viện  
pip install -r requirements.txt  
  
# 3. Chạy server  
python app.py
```