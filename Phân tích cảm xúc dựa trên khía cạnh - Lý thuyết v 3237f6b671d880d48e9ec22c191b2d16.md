# Phân tích cảm xúc dựa trên khía cạnh - Lý thuyết và ứng dụng thực tế cho Tiếng Việt

## Khái niệm

Phân tích cảm xúc dựa trên khía cạnh (Aspect-based sentiment analysis - ABSA) là một kỹ thuật phân tích chi tiết cảm xúc trong xử lý ngôn ngữ tự nhiên (NLP) nhằm trích xuất ra cực tính cảm xúc đối với các khía cạnh hoặc thuộc tính cụ thể của một đối tượng nhất định.

Ví dụ một đánh giá về sản phẩm thời trang:

> Đồ ăn ở đây ngon đấy, nhưng giá hơi cao!
> 

Chúng ta có thể xác định người dùng có cảm xúc tích cực về mặt “chất lượng thực phẩm”, nhưng có cảm xúc tiêu cực với khía cạnh “chi phí”. Điều này giúp phân biệt ABSA với các phương pháp phân tích cảm xúc truyền thống, vốn chỉ xác định được một cực tính cảm xúc duy nhất cho toàn bộ văn bản, dẫn đến việc không làm rõ được sự khác biệt trong đánh giá, cảm nhận về nhiều khía cạnh khác nhau của một đối tượng duy nhất.

Trong bài toán ABSA, có 4 yếu tố quan trọng nhất: khía cạnh được đánh giá a, danh mục khía cạnh c, quan điểm o và cực tính cảm xúc p. Mối liên hệ giữa các yếu tố được minh họa trong hình dưới đây:

![Alt text](/image/absa-element.png)

## Các tác vụ và phương pháp cho bài toán ABSA

[Zhang và cộng sự](https://arxiv.org/pdf/2203.01054) đã tổng hợp được 11 tác vụ của bài toán ABSA, trong đó có 5 tác vụ đơn (chỉ có 1 output) và 6 tác vụ kép (có hơn 1 output). Mô tả chi tiết về từng tác vụ được trình bày trong hình sau:

![Alt text](/image/absa-task.png)

Về mặt phương pháp, có thể chia thành 4 nhóm chính:

- Phương pháp dựa trên từ điển (Lexicon-based Approach): Trong cách tiếp cận này, mỗi từ trong văn bản sẽ được đối chiếu với từ điển cảm xúc, trong đó các từ được gán nhãn cảm xúc như tích cực, tiêu cực hoặc trung tính. Sau đó, các quy tắc cú pháp và ngữ nghĩa được áp dụng để xác định mối quan hệ giữa từ mang cảm xúc và khía cạnh (aspect) trong câu. Ví dụ, trong câu “Màn hình rất đẹp nhưng pin khá yếu”, các luật phụ thuộc cú pháp có thể xác định rằng từ “đẹp” liên quan đến khía cạnh “màn hình”, trong khi “yếu” liên quan đến khía cạnh “pin”.
    
    Ưu điểm của phương pháp này là dễ triển khai, không cần dữ liệu huấn luyện lớn và có tính giải thích cao. Tuy nhiên, nhược điểm là khó xử lý các hiện tượng ngôn ngữ phức tạp như mỉa mai, phủ định nhiều tầng hoặc phụ thuộc ngữ cảnh.
    
- Phương pháp học máy truyền thống (Traditional Machine Learning): Trong cách tiếp cận này, bài toán ABSA thường được chia thành các bài toán con gồm trích xuất khía cạnh (aspect extraction) và phân loại cảm xúc (sentiment classification). Các mô hình học máy bao gồm: SVM, Naive Bayes, CRF, Maximum Entropy,…
    
    Ưu điểm của phương pháp học máy là khả năng học từ dữ liệu và tổng quát tốt hơn so với các phương pháp dựa trên luật. Tuy nhiên, hiệu quả của chúng phụ thuộc nhiều vào chất lượng của đặc trưng được thiết kế thủ công, và việc xây dựng đặc trưng này thường đòi hỏi nhiều công sức và kiến thức chuyên môn về ngôn ngữ học.
    
- Phương pháp học sâu (Deep Learning): Sự phát triển của Deep Learning đã mở ra một hướng tiếp cận mới cho bài toán ABSA. Thay vì dựa vào các đặc trưng thủ công, các mô hình học sâu có khả năng tự động học biểu diễn đặc trưng từ dữ liệu thông qua các mạng nơ-ron nhiều lớp. Nó cũng có khả năng mô hình hóa ngữ cảnh của từ trong câu, từ đó xác định chính xác hơn mối quan hệ giữa aspect và sentiment expression. Ví dụ, cơ chế attention cho phép mô hình tập trung vào những từ quan trọng trong câu khi xác định cảm xúc đối với một khía cạnh cụ thể.
- Phương pháp dựa trên mô hình ngôn ngữ tiền huấn luyện (Pre-trained Language Models): Gần đây, các nghiên cứu ABSA ngày càng tập trung vào việc sử dụng các mô hình ngôn ngữ tiền huấn luyện vốn được huấn luyện trước trên tập dữ liệu văn bản rất lớn. Các mô hình này có khả năng nắm bắt tốt ngữ cảnh và các mối quan hệ ngữ nghĩa, từ đó cải thiện đáng kể hiệu suất của các tác vụ ABSA.

![Alt text](/image/absa-method.png)

## Ví dụ về một dự án ABSA

[def]: image%201.png