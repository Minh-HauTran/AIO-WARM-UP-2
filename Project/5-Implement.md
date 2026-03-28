## 5. Implementation:
### 5.1. Machine Learning Models:
#### 5.1.1. Tổng quan pipeline training:

Code được tổ chức theo hướng khá rõ ràng, gồm 4 phần chính:
1. **Tạo model** từ tên mô hình.
2. **Huấn luyện và đánh giá bằng K-Fold**.
3. **Tính metric** gồm Accuracy, Precision, Recall, F1-score.
4. **Hiển thị và so sánh kết quả** giữa các mô hình.

#### 5.1.2. Hàm tạo mô hình

Phần đầu tiên là hàm `create_ml_model(model_type, **kwargs)`. Hàm này đóng vai trò như một **factory function**: truyền vào tên model, hàm sẽ trả về đúng đối tượng mô hình tương ứng. Các model được dùng để train cho bài toàn này:
- `naive_bayes` → `MultinomialNB`
- `svm` → `LinearSVC`
- `random_forest` → `RandomForestClassifier`
- `logistic_regression` → `LogisticRegression`
- `ml_ensemble` → mô hình stacking do hàm `create_ml_stacking_classifier()` tạo ra

Ví dụ:
```python
lr_model = create_ml_model(  
    'logistic_regression',  
    max_iter=5000,  
    solver='saga',  
    multi_class='multinomial'  
)
```

Đoạn trên tạo ra một mô hình Logistic Regression cho bài toán phân loại nhiều lớp. `max_iter=5000` giúp mô hình có đủ số vòng lặp để hội tụ, còn `solver='saga'` phù hợp với dữ liệu lớn và hỗ trợ `multinomial`.

#### 5.1.3. Train models bằng Stratified K-Fold:

Phần quan trọng nhất của code nằm ở hàm `kfold_evaluation(model, X, y, kfold)`. Đây là nơi mô hình được train và đánh giá.

##### 3.1. Vì sao dùng Stratified K-Fold?

Code khởi tạo:
```python
skf = StratifiedKFold(n_splits=5, shuffle=True)
```

`StratifiedKFold` khác với KFold thường ở chỗ nó đảm bảo **tỷ lệ các lớp trong mỗi fold gần giống với toàn bộ dataset**. Điều này đặc biệt quan trọng trong các bài toán phân loại khi dữ liệu bị lệch lớp.

##### 3.2. Quy trình train trong từng fold

Trong mỗi vòng lặp:
1. Tách dữ liệu thành `X_train`, `X_val`, `y_train`, `y_val`
2. Gọi `model.fit(X_train, y_train)` để train
3. Dự đoán trên cả tập train và validation
4. Tính metric cho cả hai tập
5. Lưu lại kết quả từng fold

Code cốt lõi:
```python
for train_index, val_index in skf.split(X, y):  
    X_train, X_val = X[train_index], X[val_index]  
    y_train, y_val = y[train_index], y[val_index]  
  
    model.fit(X_train, y_train)  
  
    train_predictions = model.predict(X_train)  
    val_predictions = model.predict(X_val)
```

Ý tưởng rất chuẩn:

- **Train metrics** cho biết mô hình học tốt đến mức nào trên dữ liệu đã thấy
- **Validation metrics** cho biết mô hình tổng quát hóa tốt đến đâu trên dữ liệu chưa thấy trong fold đó

##### 3.3. Tính kết quả trung bình sau 5 folds

Sau khi chạy xong 5 folds, code lấy trung bình Accuracy, Precision, Recall và F1-score của các tập validation:

```python
model_results = {  
    "accuracy": np.mean([result["accuracy"] for result in fold_results]),  
    "precision": np.mean([result["precision"] for result in fold_results]),  
    "recall": np.mean([result["recall"] for result in fold_results]),  
    "f1": np.mean([result["f1"] for result in fold_results])  
}
```

Điều này cho ta một đánh giá ổn định hơn so với việc chỉ chia train/test một lần.

##### 3.4. Retrain trên toàn bộ dữ liệu

Cuối hàm có đoạn:
```python
model.fit(X, y)
```

Điều này nghĩa là sau khi đánh giá xong bằng cross-validation, mô hình sẽ được train lại trên **toàn bộ dataset** để tạo ra phiên bản cuối cùng sẵn sàng đem đi sử dụng hoặc lưu lại.

### 5.2. Deep Learning Models:

#### 5.2.1. Tổng quan pipeline training:

Toàn bộ pipeline Deep Learning gồm 4 bước chính:
##### Bước 1: Chuẩn bị dữ liệu
- Dữ liệu đã được **oversampling**
- Sau đó split thành:
```python
X_train, X_val, y_train, y_val = train_test_split(..., test_size=0.1
```

Chia dataset thành 2 phần: 90% train, 10% validation
##### Bước 2: Định nghĩa hyperparameter search space
```python
param_dist = {  
    'learning_rate': [0.01, 0.001, 0.0001],  
    'dense_units': [32, 64, 128],  
    'dropout': [0.2, 0.3, 0.4, 0.5],  
    'l1_reg': [0.001, 0.01, 0.1],  
    'l2_reg': [0.001, 0.01, 0.1],  
    ...  
}
```

Đây là không gian tìm kiếm hyperparameter cho model.
##### Bước 3: Hyperparameter tuning bằng RandomizedSearchCV

Hàm chính:
```python
perform_random_search(...)
```

Ý tưởng:
- Wrap model bằng `KerasClassifier`
- Random sample các config
- Train + validate bằng cross-validation
- Chọn best parameters
##### Bước 4: Evaluate model
Sau khi tìm best model:
```python
display_dl_results(best_model, ...)
```

- Predict trên validation set
- Tính:
    - Accuracy
    - Precision
    - Recall
    - F1-score
#### 5.2.2. Các mô hình được dùng để train:

##### Model 1 - Fully Connected Neural Network:

Best hyperparameters:
```
learning_rate = 0.001
dense_units = 32
dropout = 0.2
epochs = 5
batch_size = 64
```
##### Model 2: GRU

Best hyperparameters:
```
learning_rate = 0.001
dense_units = 64
dropout = 0.2
batch_size = 16
```
##### Model 3: Bidirectional LSTM

Best hyperparameters:
```
learning_rate = 0.001  
dense_units = 64  
dropout = 0.2  
epochs = 3
```
#### 5.2.3. PhoBERT Model:

Khác với các mô hình Deep Learning ở trên như Fully Connected, GRU hay Bidirectional LSTM, PhoBERT không học từ đầu hoàn toàn trên dataset của bài toán, mà sử dụng mô hình ngôn ngữ tiếng Việt đã được pretrained sẵn là `vinai/phobert-base`, sau đó fine-tune lại cho bài toán phân loại cảm xúc 3 lớp. Trong code, PhoBERT được triển khai theo pipeline:
```
Text -> Tokenizer -> input_ids + attention_mask -> PhoBERT -> CLS token -> Dense -> Softmax
```
Cách làm này cho phép mô hình tận dụng kiến thức ngôn ngữ đã học trước đó trên tập dữ liệu lớn, từ đó biểu diễn ngữ nghĩa của câu tốt hơn so với các mô hình train từ đầu.

##### Bước 1: Tokenization và mã hóa dữ liệu

Đầu tiên, code khởi tạo tokenizer của PhoBERT bằng `AutoTokenizer.from_pretrained("vinai/phobert-base")`. Sau đó, hàm `encode_texts_for_phobert(sentences, max_length=128)` được dùng để chuyển văn bản đầu vào thành hai tensor:
- `input_ids`: biểu diễn ID của các token
- `attention_mask`: đánh dấu vị trí token thật và vị trí padding

Trong quá trình encode, dữ liệu được:
- chuyển về danh sách chuỗi
- cắt bớt nếu vượt quá độ dài tối đa
- padding về cùng độ dài `max_length = 128`

Điều này giúp toàn bộ câu đầu vào có cùng kích thước khi đưa vào mô hình.
##### Bước 2: Xây dựng mô hình PhoBERT

PhoBERT được implement trong hàm `create_phobert_model(max_length=128, learning_rate=2e-5)`. Mô hình nhận hai đầu vào là `input_ids` và `attention_mask`, sau đó truyền qua backbone `TFAutoModel.from_pretrained("vinai/phobert-base")`.

Output của PhoBERT là tensor biểu diễn ngữ cảnh cho toàn bộ chuỗi. Trong code, nhóm lấy vector của token đầu tiên (`CLS token`) làm vector đại diện cho cả câu:
```python
cls_token = embeddings[:, 0, :]
```

Vector này sau đó được đưa qua:
- `Dropout(0.3)` để giảm overfitting
- `Dense(64, activation='relu')`
- `Dense(3, activation='softmax')` để dự đoán xác suất cho 3 lớp

##### Bước 3: Compile và train mô hình
Sau khi xây dựng xong, mô hình được compile với:
- `optimizer = Adam`
- `learning_rate = 2e-5`
- `loss = sparse_categorical_crossentropy`
- `metrics = ['accuracy']`

Learning rate rất nhỏ là lựa chọn phù hợp khi fine-tune các mô hình pretrained như PhoBERT, nhằm tránh làm mất đi những tri thức ngôn ngữ đã học sẵn từ trước.

Quá trình train được thực hiện bằng:
```python
history_phobert = phobert_model.fit(  
    x=[train_input_ids, train_attention_mask],  
    y=y_train,  
    validation_data=([val_input_ids, val_attention_mask], y_val),  
    epochs=1,  
    batch_size=16  
)
```

Điều này cho thấy mô hình được fine-tune trực tiếp trên tập train, đồng thời theo dõi hiệu năng trên tập validation.

##### Bước 4: Kết quả mô hình

Sau quá trình huấn luyện, PhoBERT đạt kết quả trên tập validation như sau:
- Accuracy: **93.94%**
- Precision: **0.94**
- Recall: **0.94**
- F1-score: **0.94**

Kết quả này cho thấy PhoBERT là một trong những mô hình mạnh nhất trong toàn bộ pipeline, nhờ khả năng tận dụng biểu diễn ngữ nghĩa sâu của tiếng Việt từ quá trình pretraining.
##### Các tham số chính được sử dụng khi fine-tune PhoBERT trong code gồm:
```
max_length = 128  
learning_rate = 2e-5  
dropout = 0.3  
dense_units = 64  
batch_size = 16  
epochs = 1
```