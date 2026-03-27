## 4. Data Preparation & Feature Engineering:

Trong bài toán này, nhóm sẽ sử dụng dataset: `uit-nlp/vietnamese_students_feedback` (HuggingFace)
Dataset bao gồm 16,175 samples, chứa các phản hồi của sinh viên về giảng viên, bao gồm:

- Nội dung văn bản (Sentence)
- Nhãn cảm xúc (Sentiment) (0: Negative, 1: Neutral, 2: Positive)
- Chủ đề (Topic)

### 4.1. Data Prepareation

Dataset được lấy từ HuggingFace:

```python
dataset = load_dataset("uit-nlp/vietnamese_students_feedback")
```

Sau đó, tiến hành trộn các tập dataset (train + validation + test) thành 1 DataFrame để thuận tiện cho việc xử lý:

```python
df = pd.concat([train_df, val_df, test_df], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)
df.rename(columns={'sentence': 'content', 'sentiment': 'label'}, inplace=True)
```

### 4.2. Text Processing:

Trước tiên, nhóm sẽ tiền hành bước làm sạch dữ liệu bằng cách loại bỏ những dữ liệu bị lăp, và những dữ liệu mang giá trị NaN:

```python
df = df.drop_duplicates("content")  # remove duplicates
df = df.dropna()                   # remove null values
```

Tiếp theo, nhóm sẽ bắt đầu tiến hành việc tiền xử lý văn bản:

1. Chuẩn hóa văn bản
2. Loại bỏ emoji
3. Chuẩn hóa kí tự lặp
4. Chuẩn hóa dấu câu
5. Loại bỏ dấu câu & kí tự đặc biệt
6. Chuẩn hóa khoản trắng

Cuối cùng, nhóm sẽ tiến hành tokenization tiếng Việt, và kết quả:

| STT | Content                                           | Label | Topic | Corpus                                             |
| --- | ------------------------------------------------- | ----- | ----- | -------------------------------------------------- |
| 0   | tổ chức các cuộc thi liên quan tới kỹ năng gia... | 2     | 1     | tổ_chức các cuộc thi liên_quan tới kỹ_năng gia...  |
| 1   | ít mục không đi sâu vào .                         | 0     | 3     | ít mục không đi_sâu vào                            |
| 2   | thầy cung cấp nhiều kiến thức mới , thầy dạy t... | 2     | 0     | thầy cung_cấp nhiều kiến_thức mới thầy dạy tận...  |
| 3   | boss cuối , thầy dạy quá hay , cả kiến thức mô... | 2     | 0     | bos cuối thầy dạy quá hay cả kiến_thức môn_học...  |
| 4   | thấy rất giỏi và dạy rất tốt , nhiệt tình , dễ... | 2     | 0     | thấy rất giỏi và dạy rất tốt nhiệt_tình dễ hiểu    |
| 5   | thầy wzjwz307 dạy giỏi từ xưa đến giờ .           | 2     | 0     | thầy wzjwz307 dạy giỏi từ xưa đến giờ              |
| 6   | giảng viên tuyệt vời nhất uit .                   | 2     | 0     | giảng_viên tuyệt_vời nhất uit                      |
| 7   | giáo viên dạy có tâm huyết , nhiệt tình với si... | 2     | 0     | giáo_viên dạy có tâm_huyết nhiệt_tình với sinh...  |
| 8   | một số phần thầy chưa nói rõ gây khó khăn cho ... | 0     | 0     | một_số phần thầy chưa nói rõ gây khó_khăn cho ...  |
| 9   | chưa thực sự tận dụng tốt thời gian và chưa mở... | 2     | 0     | chưa thực_sự tận_dụng tốt thời_gian và chưa mở...  |
| 10  | em thấy việc thầy dời deadline nó không phù hợ... | 0     | 0     | em thấy việc thầy dời deadline nó không phù_hợp... |
| 11  | dễ tiếp cận kiến thức .                           | 2     | 0     | dễ tiếp_cận kiến_thức                              |
| 12  | thầy nên đưa ra những ví dụ trong khi giảng bà... | 0     | 0     | thầy nên đưa ra những ví_dụ trong khi giảng bà...  |
| 13  | thay vì ngồi chờ chấm điểm như đi thi thực hành . | 1     | 1     | thay_vì ngồi chờ chấm điểm như đi thi thực_hành    |
| 14  | thầy cố gắng giảng bài cho mọi người hiểu , cố... | 2     | 0     | thầy cố_gắng giảng_bài cho mọi người hiểu cố_g...  |
| 15  | giảng bài dễ hiểu , dễ vận dụng .                 | 2     | 0     | giảng bài dễ hiểu dễ vận_dụng                      |

### 4.3. EDA:

#### 4.3.1. Phân tích tần suất từ (Bag-of-Words):

Đầu tiên, nhóm sẽ xây dựng một biểu diễn đơn giản dạng Bag-of-Words bằng cách gom toàn bộ các câu đã được xử lý (corpus) và đếm tấn suất xuất hiện của token:

```python
all_words = [token for token in df['corpus'].tolist() if token and token != '']
corpus = ' '.join(all_words)
all_words = nltk.FreqDist(all_words)
```

Kết quả:

```
Number of words: 16001
Most common words: [('thầy dạy hay dễ hiểu', 4), ('nhiệt_tình vui_tính', 3), ('thầy dạy nhiệt_tình tận_tâm', 3), ('giảng_viên dạy nhiệt_tình dễ hiểu', 3), ('giảng_viên tận_tâm nhiệt_tình', 3), ('nhiệt_tình tâm_huyết', 3), ('giảng_viên nhiệt_tình tận_tâm', 3), ('em cảm_ơn', 3), ('thầy giảng bài dễ hiểu nhiệt_tình', 2), ('cô dạy rất hay', 2), ('nhiệt_tình thân_thiện', 2), ('cô nhiệt_tình tận_tâm', 2), ('em rất thích', 2), ('thầy tận tâm', 2), ('vui_vẻ nhiệt_tình', 2)]
```

#### 4.3.2. WordCloud Visualization:

Để trực quan hóa tàn suất từ, nhóm sử dụng WordCloud.
Ý nghĩa của biểu đồ:

- Từ xuất hiện càng nhiều - hiển thị càng lớn
- Giúp nhanh chóng nhận diện: chủ đề - tính chất của feedback

Code như sau:

```python
word_cloud = wordcloud.WordCloud(
    max_words=100,
    background_color="black",
    width=2000,
    height=1000
).generate(corpus)
```

Kết quả:
![Hình 4.1: WordCloud](img/4-Hinh1.png)

#### 4.3.3. Phân phối độ dài câu:

Nhóm phân tích độ dài của từng câu trong bộ dữ liệu, việc này giúp chúng ta hiểu được:

- Số lượng câu ngắn, câu dài
- Có cần padding/ truncation không?

Code như sau:

```python
# Calculate the length of each sentence directly
lengths = df['content'].apply(len)

# Plot histogram
plt.figure(figsize=(10, 5))
plt.hist(lengths, bins=30, edgecolor='k', alpha=0.7)
plt.title('Distribution of sentence lengths in Customer Reviews')
plt.xlabel('Sentence Length')
plt.ylabel('Number of Sentences')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()
```

Kết quả:
![Hình 4.2: Trực quan hóa độ dài câu](img/4-Hinh2.png)

#### 4.3.3. Phân phối nhãn và topic:

Sau bước tiền xử lý và khám phá dữ liệu ban đầu, chúng tôi tiến hành phân tích phân phối của **sentiment labels** và **topics** nhằm hiểu rõ hơn về cấu trúc dataset. Đây là bước quan trọng giúp phát hiện các vấn đề như mất cân bằng dữ liệu (data imbalance), từ đó định hướng chiến lược huấn luyện mô hình.

Kết quả sau khi trực quan hóa:
![Hình 4.3: Trực quan hóa nhãn và topic](img/4-Hinh3.png)

##### Phân phối nhãn cảm xúc:

Biểu đồ tròn thể hiện tỷ lệ các nhãn cảm xúc trong tập dữ liệu:

- Positive (~49.7%)
- Negative (~46.0%)
- Neutral (~4.3%)
  Từ đó, ta thấy rằng dataset cân bằng giữa hai lớp Positive và Negative, nhưng lớp Neutral chiếm tỷ lệ rất nhỏ

##### Phân phối chủ đề:

Biểu đồ cột dưới đây thể hiện số lượng review theo từng chủ đề:

- **Lecturer** chiếm đa số (áp đảo)
- **Training Program** đứng thứ hai
- **Facility** và **Others** có số lượng khá ít

### 4.4. Over-sampling:

Trong bước phân tích trước, nhóm nhận thấy rằng dataset gặp vấn đề **mất cân bằng dữ liệu (class imbalance)**, đặc biệt ở lớp **Neutral**, khi chỉ chiếm khoảng ~4% tổng số mẫu.

Để giải quyết vấn đề này, chúng tôi áp dụng kỹ thuật **Over-sampling** nhằm cân bằng lại phân phối các nhãn.
Cụ thể:

- Lớp **Neutral (label = 1)** là lớp thiểu số
- Chúng tôi sẽ **duplicate các sample Neutral** cho đến khi số lượng gần bằng các lớp còn lại

Cách thực hiện:

```python
neutral_indices = np.where(train_labels == 1)[0]
oversample_size = len(train_labels[train_labels == 0]) - len(neutral_indices)

oversampled_neutral_indices = resample(
    neutral_indices,
    replace=True,
    n_samples=oversample_size
)
```

Sau Oversampling, có một bộ dataset mới chứa dữ liệu gốc, thêm dữ liệu Neutral nhân bản.
Kết quả:

```
Label 0: 6695
Label 1: 6695
Label 2: 7233
```

![Hình 4.4: Trực quan dataset sau khi Oversampling](img/4-Hinh4.png)

Bước này cực kì quan trọng, để làm cho dataset được cân bằng hơn, việc này giúp cho model có thể học tốt hơn

### 4.5. Feature Engineering:

#### 4.5. Bag-of-Words:

Sử dụng `CountVectorizer` để chuyển văn bản thành vector số bằng cách **đếm số lần xuất hiện của từng từ** trong câu.

Triển khai:

```python
vectorizer = CountVectorizer()
vectorizer.fit(train_sentences)
```

Kết quả:

```
Sentence: bài giảng dễ hiểu
Token : Count
bài : 1
dễ : 1
giảng : 1
hiểu : 1
------

Sentence: giảng_viên hỗ_trợ tích_cực
Token : Count
giảng_viên : 1
hỗ_trợ : 1
tích_cực : 1
------

Sentence: giảng_dạy nhiệt_tình đúng giờ
Token : Count
giảng_dạy : 1
giờ : 1
nhiệt_tình : 1
đúng : 1
------
```

Áp dụng cho toàn bộ dữ liệu:

```python
X_oversampled_bow = vectorizer.transform(train_sentences_oversampled)
y_oversampled_bow = train_labels_oversampled
```

#### 4.6. Text Vectorization:

Sử dụng `TextVectorization` để chuyển văn bản thành **chuỗi số (integer sequence)**, phục vụ cho các mô hình deep learning.

#### Cấu hình:

```python
MAX_VOCAB_LENGTH = 20000

sequence_lengths = [len(sentence.split()) for sentence in train_sentences]
MAX_LENGTH = int(np.percentile(sequence_lengths, 95))
```

- `MAX_VOCAB_LENGTH`: giới hạn số từ trong vocabulary
- `MAX_LENGTH`: độ dài câu (lấy theo percentile 95% để tránh outliers)

#### Khởi tạo:

```python
text_vectorizer = TextVectorization(
    max_tokens=MAX_VOCAB_LENGTH,
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    output_mode="int",
    output_sequence_length=MAX_LENGTH
)
```

#### Huấn luyện Vocabulary:

```python
text_vectorizer.adapt(train_sentences)
```

#### Kết quả:

```
Number of words in vocab: 4543
Top 5 most common words: ['', '[UNK]', 'thầy', 'sinhviên', 'dạy']
Bottom 5 least common words: ['2000', '200', '1983', '19', '140']
```

#### 4.7. Embedding:

Embedding là lớp dùng để chuyển các số (sau khi vectorization) thành **vector dense có ý nghĩa ngữ nghĩa**.

#### Khởi tạo:

```python
def create_embedding_layer(
    input_dim=MAX_VOCAB_LENGTH,
    output_dim=128,
    input_length=MAX_LENGTH
):
    return layers.Embedding(
        input_dim=input_dim,
        output_dim=output_dim,
        input_length=input_length
    )
```

- `input_dim`: kích thước vocabulary
- `output_dim`: số chiều của vector embedding (ví dụ: 128)
- `input_length`: độ dài chuỗi input

#### Cách hoạt động:

Input:

```
[12, 45, 78, ...]
```

Output:

```
[
  [0.12, -0.45, ...],
  [0.78, 0.11, ...],
  ...
]
```

### 4.7. Pipeline Summary:

![Pipeline](4-Hinh5.png)
