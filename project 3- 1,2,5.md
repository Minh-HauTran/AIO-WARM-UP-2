Kiến trúc Hệ thống Phân tích Cảm xúc Phản hồi của Sinh viên Việt Nam (Vietnamese Student Sentiment Analysis)

## 1. Introduction

### 1.1. Giới thiệu bài toán

Trong bối cảnh toàn cầu hóa và sự cạnh tranh ngày càng gay gắt giữa các cơ sở giáo dục đại học, việc đảm bảo và nâng cao chất lượng đào tạo đã trở thành mục tiêu chiến lược mang tính sống còn. Theo truyền thống, các trường đại học thường đánh giá chất lượng giảng dạy thông qua các hệ thống khảo sát định lượng định kỳ vào cuối mỗi học kỳ. Mặc dù các thang điểm đánh giá (như thang Likert) cung cấp một cái nhìn tổng quan về mức độ hài lòng của sinh viên, chúng lại thiếu đi chiều sâu và sự thấu cảm cần thiết để hiểu rõ nguyên nhân gốc rễ của các vấn đề phát sinh trong môi trường sư phạm. Ngược lại, những lời nhận xét định tính (phản hồi dạng văn bản tự do) lại chứa đựng nguồn thông tin phong phú, phản ánh chi tiết những mong muốn, sự thất vọng, hay sự trân trọng của sinh viên đối với phương pháp giảng dạy, thái độ của giảng viên, nội dung chương trình học, và điều kiện cơ sở vật chất.

Tuy nhiên, khối lượng dữ liệu phản hồi dạng văn bản này thường đạt đến quy mô hàng chục ngàn, thậm chí hàng trăm ngàn mẫu mỗi năm tại các trường đại học lớn. Việc đọc, phân loại và trích xuất ý nghĩa từ lượng dữ liệu khổng lồ này bằng phương pháp thủ công không chỉ tiêu tốn một lượng lớn thời gian và nguồn lực hành chính mà còn tiềm ẩn nguy cơ sai lệch do những định kiến chủ quan của con người. Đứng trước thách thức này, sự ra đời của các kỹ thuật Khai phá Dữ liệu Văn bản (Text Mining) và Phân tích Cảm xúc (Sentiment Analysis) đã mở ra một hướng giải quyết triệt để và hiệu quả. Bài toán "Vietnamese Student Sentiment Analysis" được khởi xướng nhằm mục đích cốt lõi là xây dựng một hệ thống Trí tuệ Nhân tạo (Artificial Intelligence - AI) có khả năng tự động đọc hiểu, phân tích, và phân loại các sắc thái cảm xúc từ những lời nhận xét bằng tiếng Việt của sinh viên. Hệ thống này được kỳ vọng sẽ đóng vai trò như một màng lọc thông minh, hỗ trợ ban lãnh đạo nhà trường và các giảng viên có được những báo cáo phân tích theo thời gian thực về mức độ thỏa mãn của người học, từ đó đưa ra các quyết định điều chỉnh kịp thời, thúc đẩy mô hình cải tiến liên tục (Continuous Improvement Process) trong giáo dục.

### 1.2. Mô hình hóa bài toán: Input và Output

Để máy tính có thể xử lý và giải quyết bài toán phân tích cảm xúc, quy trình ngôn ngữ học tự nhiên phức tạp cần được trừu tượng hóa và chuyển đổi thành một bài toán học thống kê. Dưới góc độ toán học và kiến trúc học máy, quá trình này được định nghĩa là một bài toán Phân loại Cấp độ Câu (Sentence-level Classification) với cấu trúc thiết kế tập trung vào việc tìm ra một hàm ánh xạ tối ưu.

Quá trình ánh xạ này được chia thành các không gian đầu vào và đầu ra cụ thể như sau. Về mặt đầu vào (Input), hệ thống tiếp nhận một tập hợp các chuỗi ký tự không có cấu trúc toán học rõ ràng, được biểu diễn dưới dạng ngôn ngữ tự nhiên tiếng Việt. Mỗi mẫu dữ liệu đầu vào có thể là một câu đơn, một câu ghép, hoặc một đoạn văn bản ngắn chứa đựng ý kiến của sinh viên. Bản chất của không gian đầu vào này vô cùng phức tạp do đặc thù của tiếng Việt là ngôn ngữ đơn lập, sử dụng thanh điệu, hệ thống từ vựng bao gồm một lượng lớn các từ ghép, từ láy, từ mượn, cùng với vô vàn các biến thể ngôn ngữ mạng, lỗi chính tả, và biểu tượng cảm xúc (emoji). Hệ thống cần phải có một cơ chế mã hóa (encoding mechanism) để biến đổi các chuỗi ký tự này thành các véc-tơ số học nhiều chiều (high-dimensional numerical vectors) mà mạng nơ-ron có thể thực hiện các phép toán đại số tuyến tính.

Sau khi tín hiệu đầu vào đi qua hệ thống hàm ánh xạ (được cấu thành từ hàng triệu đến hàng trăm triệu tham số tối ưu hóa), đầu ra (Output) được định hình là một véc-tơ phân phối xác suất dự đoán trạng thái cảm xúc của văn bản đó. Không gian nhãn (Label Space) của dự án này được thiết kế theo một cấu trúc rời rạc bao gồm ba phân lớp chính:

- **Tích cực (Positive)**: Các phản hồi mang ý nghĩa khen ngợi, động viên, thể hiện sự hài lòng cao về phương pháp sư phạm của giảng viên, tài liệu môn học, hoặc môi trường học tập. Những nhận xét này thường chứa các từ ngữ biểu cảm mạnh như "tuyệt vời", "nhiệt tình", "dễ hiểu", "tận tâm".

- **Trung lập (Neutral)**: Những câu văn mang tính chất trần thuật khách quan, các ý kiến đóng góp mang tính xây dựng nhưng không biểu lộ trạng thái cảm xúc vui buồn rõ rệt, hoặc những phản hồi chỉ đơn thuần xác nhận thông tin (ví dụ: "môn học có nhiều bài tập", "thầy sử dụng slide tiếng Anh").

- **Tiêu cực (Negative)**: Các phản hồi chê trách, phàn nàn, phê phán, hoặc thể hiện sự không hài lòng, thất vọng của sinh viên về bất kỳ khía cạnh nào của quá trình đào tạo. Nhóm này thường xuất hiện các từ khóa như "nhàm chán", "buồn ngủ", "khó hiểu", "đi trễ".

Mục tiêu tối thượng của mô hình là tối đa hóa độ chính xác dự đoán đầu ra này so với nhãn thực tế (ground truth) thông qua việc tinh chỉnh các trọng số bên trong quá trình huấn luyện, giảm thiểu hàm mất mát (loss function) trên toàn bộ không gian dữ liệu phân phối.

## 2. Ý nghĩa, Các Liên quan và Sự Tiến hóa của Cấu trúc Thuật toán

### 2.1. Sơ đồ Cấu trúc Đánh giá Chất lượng và Sự Tiến hóa của Trí tuệ Nhân tạo

Sự giao thoa giữa Khoa học Quản lý và Khoa học Máy tính được thể hiện một cách mạch lạc thông qua lăng kính của quá trình đánh giá chất lượng. Trong lý thuyết quản trị, các phản hồi của sinh viên chính là dữ liệu nền tảng để đo lường Chất lượng Dịch vụ (Service Quality) hoặc Chất lượng Hệ thống (System Quality) – được gọi tắt chung là SQ. Sự thỏa mãn của người học (Student Satisfaction) tỷ lệ thuận với mức độ đáp ứng của SQ. Để xử lý khối lượng dữ liệu SQ khổng lồ nhằm rút ra những kết luận chiến lược, các học giả và kỹ sư đã áp dụng các tiến bộ của Trí tuệ Nhân tạo theo một hệ thống phân cấp ngày càng tinh vi và phức tạp.

Cấu trúc dòng chảy từ dữ liệu quản trị đến các cơ chế xử lý tính toán hiện đại được minh họa thông qua sơ đồ sau:

![Hình 1: Sơ đồ luồng ứng dụng từ Dữ liệu Chất lượng hệ thống (SQ) qua các cấp độ tiến hóa của Trí tuệ Nhân tạo trong phân tích dữ liệu]

**Hình 1:** Sơ đồ luồng ứng dụng từ Dữ liệu Chất lượng hệ thống (SQ) qua các cấp độ tiến hóa của Trí tuệ Nhân tạo trong phân tích dữ liệu.

Sơ đồ trên trình bày sự phát triển theo chiều sâu của các cơ chế luận giải dữ liệu. Dữ liệu khởi nguồn (SQ) là những tín hiệu ngôn ngữ tự nhiên phi cấu trúc. Để giải mã chúng, con người cần đến Trí tuệ Nhân tạo (AI) - khái niệm bao trùm ám chỉ bất kỳ hệ thống máy tính nào có khả năng mô phỏng nhận thức con người. Đi sâu vào bên trong AI, Học máy (Machine Learning - ML) cung cấp các thuật toán học thống kê, cho phép hệ thống nhận diện cảm xúc dựa trên tần suất xuất hiện của từ vựng thông qua các hàm toán học tối ưu hóa. Mở rộng từ ML, Học sâu (Deep Learning - DL) ứng dụng các mạng nơ-ron đa lớp phức tạp để tự động biểu diễn và nắm bắt ngữ cảnh không gian của văn bản mà không cần con người định nghĩa các đặc trưng thủ công. Cuối cùng, ở mức độ phức tạp nhất trong sơ đồ là Học tăng cường (Reinforcement Learning - RL), một nhánh tối ưu hóa các chuỗi quyết định dựa trên hàm phần thưởng, đang bắt đầu được nghiên cứu để tích hợp vào các hệ thống hiểu ngôn ngữ đa tác vụ.

### 2.2. Phân tích Ưu và Nhược điểm: Học Máy (ML), Học Sâu (DL) và Học Tăng cường (RL)

Việc lựa chọn kiến trúc thuật toán để giải quyết bài toán phân tích cảm xúc (Sentiment Analysis) không phải là một bài toán có duy nhất một lời giải đúng, mà đòi hỏi sự cân nhắc kỹ lưỡng giữa khả năng biểu diễn ngữ nghĩa, giới hạn về tài nguyên tính toán, và yêu cầu về độ chính xác. Bảng dưới đây cung cấp một phân tích chuyên sâu về ưu điểm và nhược điểm của ba hệ tư tưởng tính toán này trong bối cảnh Xử lý Ngôn ngữ Tự nhiên.

| Hệ thống Thuật toán          | Cơ sở Lý thuyết và Ưu điểm Cốt lõi                                                                                                                                                                                                 | Nhược điểm và Giới hạn Kỹ thuật                                                                                                                                                                                                                                                                                                                                 |
|------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Machine Learning (ML)**   | Các thuật toán ML như Logistic Regression, Support Vector Machine (SVM), hay Naive Bayes hoạt động dựa trên các nguyên lý xác suất thống kê và đại số tuyến tính nhằm tìm ra ranh giới quyết định (decision boundaries). Ưu điểm tuyệt đối của ML là thời gian huấn luyện cực kỳ nhanh chóng và yêu cầu tài nguyên phần cứng (RAM, CPU) rất thấp. ML hoạt động vô cùng ổn định trên các tập dữ liệu có kích thước nhỏ đến trung bình. Ngoài ra, tính minh bạch của ML rất cao (white-box models); các nhà nghiên cứu có thể dễ dàng lý giải tại sao mô hình lại đưa ra quyết định dựa trên trọng số của từng từ vựng cụ thể. | Nhược điểm lớn nhất của ML là phụ thuộc hoàn toàn vào kỹ nghệ trích chọn đặc trưng thủ công (Feature Engineering) như Bag-of-Words hay TF-IDF. Các biểu diễn này hoàn toàn phớt lờ trật tự từ vựng và ngữ pháp của câu. Do đó, ML thường thất bại thảm hại trước các cấu trúc ngôn ngữ phức tạp như mỉa mai, ẩn dụ, hoặc các câu có chứa từ phủ định kép. ML không thể hiểu được sự khác biệt giữa "không hay nhưng dễ hiểu" và "hay nhưng không dễ hiểu". |
| **Deep Learning (DL)**      | DL giải quyết triệt để điểm yếu của ML thông qua khả năng tự động trích xuất đặc trưng (automatic feature extraction) bằng các mạng nơ-ron nhân tạo đa lớp. Các kiến trúc như RNN, GRU, LSTM và Transformer sở hữu bộ nhớ trạng thái hoặc cơ chế tự chú ý (Self-Attention), cho phép chúng nắm bắt được mối quan hệ phụ thuộc xa (long-term dependencies) giữa các từ trong câu. DL hiểu được ngữ cảnh hai chiều của văn bản, từ đó ánh xạ chính xác các sắc thái biểu cảm phức tạp, đặc biệt là sự đa nghĩa trong tiếng Việt. Khả năng tổng quát hóa của DL trên dữ liệu lớn là vô song. | Các mô hình DL yêu cầu một lượng dữ liệu được gán nhãn khổng lồ để hội tụ mà không bị quá khớp (overfitting). Quá trình huấn luyện đòi hỏi hệ thống máy tính có hiệu năng cao trang bị Bộ xử lý Đồ họa (GPU) hoặc Bộ xử lý Tensor (TPU), dẫn đến chi phí vận hành và tiêu thụ năng lượng cực lớn. Hơn nữa, DL hoạt động như một "hộp đen" (black box), khiến việc diễn giải nguyên nhân đằng sau một dự đoán cụ thể trở nên vô cùng khó khăn, gây ra những e ngại về tính khả giải trong các ứng dụng mang tính quy phạm. |
| **Reinforcement Learning (RL)** | Khác biệt hoàn toàn với tư duy học có giám sát của ML và DL, RL học cách đưa ra các chuỗi hành động thông qua việc tương tác trực tiếp với môi trường để tối đa hóa một tín hiệu phần thưởng (reward signal) tích lũy. Ưu điểm của RL là khả năng tìm ra các chiến lược tối ưu toàn cục mà không cần một tập dữ liệu nhãn tĩnh hoàn hảo. Trong NLP, RL đặc biệt xuất sắc ở các tác vụ tạo sinh văn bản (Text Generation), tóm tắt văn bản, và hệ thống đối thoại (Chatbots), nơi mà đầu ra không phải là một nhãn duy nhất mà là một chuỗi các từ vựng hợp lý. | Khi áp dụng vào bài toán phân loại tĩnh như Sentiment Analysis, RL mang lại sự phức tạp không cần thiết. Quá trình thiết kế hàm phần thưởng và trạng thái cho bài toán phân loại rất khiên cưỡng, dẫn đến việc mô hình RL cực kỳ khó hội tụ, mất quá nhiều thời gian thử-sai, và dễ rơi vào tình trạng bất ổn định (instability). RL không phải là sự lựa chọn tối ưu, cả về mặt lý thuyết lẫn thực hành, cho các tác vụ phân loại nhãn cảm xúc đơn thuần. |

### 2.3. Lý do Lựa chọn Robofort (Random Forest) làm Đường cơ sở Nâng cao

Trong suốt quá trình triển khai dự án, để thiết lập một thang đo chuẩn mực (benchmark) nhằm đánh giá hiệu năng của các mô hình học sâu, nhóm nghiên cứu cần một thuật toán Học máy quần thể (Ensemble Machine Learning) đủ mạnh mẽ. Thuật toán Robofort (một tên gọi thay thế/viết tắt cấu trúc nội bộ của hệ thuật toán Random Forest Classifier được đề cập trong biên bản thực nghiệm dự án) đã được lựa chọn để hoàn thành vai trò này.

Sự lựa chọn Robofort (Random Forest) không phải là ngẫu nhiên mà dựa trên nền tảng lý thuyết thống kê vững chắc. Robofort hoạt động dựa trên cơ chế kết hợp hàng trăm, thậm chí hàng ngàn cây quyết định (Decision Trees) nhỏ lẻ được xây dựng trên các tập con dữ liệu ngẫu nhiên (Bootstrapping) và lựa chọn đặc trưng ngẫu nhiên. Trong bài toán phân tích cảm xúc, văn bản thường chứa một lượng lớn nhiễu (noise) từ lỗi chính tả và các từ vựng hiếm gặp. Các thuật toán tuyến tính đơn giản như Logistic Regression rất dễ bị ảnh hưởng bởi các giá trị ngoại lai này. Ngược lại, kiến trúc quần thể của Robofort giúp triệt tiêu phương sai dự đoán (variance reduction), ngăn chặn hiệu quả hiện tượng quá khớp (overfitting) thường thấy ở các cây quyết định đơn lẻ. Hơn nữa, Robofort có khả năng nắm bắt được các mối quan hệ phi tuyến tính phức tạp giữa các cụm từ (n-grams) mà không cần phải trải qua quá trình tinh chỉnh siêu tham số (hyperparameter tuning) quá khắc nghiệt. Việc Robofort đạt được độ chính xác xấp xỉ 93.67% trên tập dữ liệu UIT-VSFC chứng tỏ nó là một đối trọng thực sự đáng gờm, vạch ra giới hạn tối đa mà các phương pháp phân tích dựa trên tần suất từ vựng có thể vươn tới trước khi nhường bước cho các kiến trúc mạng nơ-ron thấu hiểu ngữ cảnh.

## 3. Phân tích Khám phá và Tiền xử lý Dữ liệu (Processing Data - EDA)

Để bất kỳ mô hình học máy hay học sâu nào có thể hoạt động hiệu quả, chất lượng của dữ liệu đầu vào đóng vai trò quyết định. Quá trình Phân tích Khám phá Dữ liệu (Exploratory Data Analysis - EDA) và Tiền xử lý được thực hiện một cách tỉ mỉ nhằm biến đổi các chuỗi văn bản thô sơ thành các cấu trúc toán học quy chuẩn.

### 3.1. Tổng quan Dữ liệu UIT-VSFC và Hiện tượng Bất đối xứng Nhãn

Dữ liệu được khai thác cho dự án này là Vietnamese Students’ Feedback Corpus (UIT-VSFC), một bộ ngữ liệu chuẩn mực, quy mô lớn, và được đánh giá cao trong giới nghiên cứu Xử lý Ngôn ngữ Tự nhiên tại Việt Nam. Bộ ngữ liệu này được các chuyên gia ngôn ngữ học gán nhãn thủ công với độ đồng thuận liên người đánh giá (inter-annotator agreement) vượt trên 91%, đảm bảo chất lượng nhãn dán ở mức cực kỳ đáng tin cậy.

Theo các phân tích thống kê trích xuất từ quá trình mã hóa dự án, tổng thể tập dữ liệu bao gồm 16.175 câu đánh giá riêng biệt. Quá trình EDA chỉ ra sự phân phối của các nhãn cảm xúc như sau:

- **Nhãn 2 (Tích cực - Positive)**: Chiếm ưu thế tuyệt đối với 8.038 mẫu, phản ánh một thực tế tâm lý học giáo dục rằng phần lớn sinh viên có xu hướng để lại những lời khen ngợi hoặc phản hồi tốt đối với các khóa học được tổ chức bài bản.

- **Nhãn 0 (Tiêu cực - Negative)**: Theo sát phía sau với 7.439 mẫu (giảm xuống còn 7.438 mẫu sau khi loại bỏ dữ liệu trùng lặp). Đây là nhóm dữ liệu mang lại giá trị thực tiễn cao nhất, giúp nhà trường phát hiện những lỗ hổng trong công tác quản lý và giảng dạy.

- **Nhãn 1 (Trung lập - Neutral)**: Là lớp thiểu số (minority class) với chỉ 698 mẫu, chiếm tỷ trọng chưa đến 5% tổng thể cơ sở dữ liệu.

Sự chênh lệch nghiêm trọng về số lượng giữa lớp Trung lập so với hai lớp còn lại tạo ra một hiện tượng được gọi là Bất đối xứng Phân phối Lớp (Class Imbalance). Dưới góc độ học máy, nếu đưa toàn bộ khối lượng dữ liệu này vào huấn luyện mà không có sự can thiệp, mô hình hàm mất mát sẽ bị chi phối hoàn toàn bởi các lớp đa số (Tích cực và Tiêu cực), dẫn đến việc mô hình sẽ "học vẹt" cách bỏ qua việc dự đoán nhãn Trung lập để giảm thiểu sai số trung bình. Do đó, kỹ thuật Over-sampling (Lấy mẫu vượt mức) đã được áp dụng trong quá trình tiền xử lý, nhân bản có chọn lọc các mẫu thuộc nhãn Trung lập để nâng tổng số lượng của chúng lên ngang bằng với các nhãn còn lại, tạo ra một không gian học tập công bằng cho mạng nơ-ron.

Bên cạnh phân phối cảm xúc, bộ dữ liệu còn cung cấp thông tin về 4 khía cạnh chủ đề (Topic) bao gồm Giảng viên (Topic 0: 11.607 mẫu), Chương trình Đào tạo (Topic 1: 3.040 mẫu), Cơ sở vật chất (Topic 2: 712 mẫu) và Các yếu tố khác (Topic 3: 816 mẫu), minh chứng cho sự đa dạng trong nội dung phản hồi của người học.

### 3.2. Đường ống Tiền xử lý Ngôn ngữ Học (Data Preprocessing Pipeline)

Tiếng Việt là một ngôn ngữ phức tạp, nơi ranh giới từ không được xác định rõ ràng chỉ bằng khoảng trắng (whitespaces) như các ngôn ngữ hệ Ấn-Âu, mà còn phụ thuộc vào ngữ nghĩa của các âm tiết đứng cạnh nhau tạo thành từ ghép. Để các mô hình có thể giải mã cấu trúc này, văn bản phải trải qua một đường ống tiền xử lý (Preprocessing Pipeline) khắt khe gồm 4 giai đoạn cốt lõi:

**Giai đoạn 1: Làm sạch Văn bản Cơ bản (Basic Text Cleaning)**  
Dữ liệu sinh viên phản hồi trên các nền tảng trực tuyến thường chứa rất nhiều nhiễu. Quá trình làm sạch khởi đầu bằng việc chuyển đổi toàn bộ văn bản về chữ in thường (lowercasing) để đồng nhất không gian véc-tơ (ví dụ: "Thầy" và "thầy" sẽ được biểu diễn như nhau). Tiếp đó, hệ thống sử dụng biểu thức chính quy (Regex) để truy quét và loại bỏ các dấu câu, ký tự đặc biệt, và một bộ lọc riêng biệt dành để tước bỏ hệ thống mã Unicode biểu diễn các biểu tượng cảm xúc (emoji) phức tạp. Một thao tác vô cùng quan trọng đối với văn phong người trẻ là việc rút gọn các ký tự kéo dài mang tính cường điệu cảm xúc (character reduction); những chuỗi ký tự bất thường như "qáaaaa" hay "hayyyy" sẽ được thuật toán nén lại thành dạng nguyên bản "qá" và "hay" nhằm tránh hiện tượng bùng nổ kích thước từ điển (vocabulary explosion).

**Giai đoạn 2: Chuẩn hóa Ngữ âm Tiếng Việt (Vietnamese Normalization)**  
Trong hệ thống gõ tiếng Việt, vị trí đặt dấu thanh điệu có thể khác nhau tùy thuộc vào bộ gõ (ví dụ: "hoà" và "hòa", "thuỷ" và "thủy"). Để giải quyết triệt để sự phân mảnh này, thư viện underthesea với hàm text_normalize được tích hợp vào luồng xử lý, có nhiệm vụ quét qua toàn bộ cơ sở dữ liệu và căn chỉnh các dấu thanh điệu về một quy chuẩn duy nhất, giúp các thuật toán học máy thống kê tần suất từ một cách chính xác tuyệt đối.

**Giai đoạn 3: Tách Từ (Word Tokenization)**  
Đây là bước có tính quyết định nhất đến việc thấu hiểu ngữ nghĩa văn bản. Nếu áp dụng các phương pháp tách từ tiếng Anh, cụm "sinh viên" sẽ bị xé lẻ thành hai token "sinh" và "viên", làm mất hoàn toàn ý nghĩa nguyên thủy của nó. Để khắc phục, dự án sử dụng hàm word_tokenize kết hợp tham số format="text" từ thư viện underthesea. Cơ chế này hoạt động dựa trên từ điển ngôn ngữ học và các mô hình Markov ẩn, tự động phát hiện các cụm từ có ý nghĩa và liên kết chúng lại bằng dấu gạch dưới (underscore). Kết quả là, những khái niệm như "sinh_viên", "giảng_viên", "bài_tập" được hệ thống nhận diện và đối xử như một đơn vị từ vựng duy nhất (single token), bảo toàn tính toàn vẹn ngữ nghĩa của cấu trúc câu.

**Giai đoạn 4: Véc-tơ hóa và Mã hóa cho Transformer (Vectorization & Tokenization)**  
Đối với các mô hình Học máy cơ sở, văn bản sau khi tách từ sẽ đi qua các lớp nhúng như CountVectorizer hoặc TextVectorization để tạo ra các ma trận đếm tần suất. Tuy nhiên, đối với kiến trúc Transformer tinh vi như PhoBERT, hệ thống sử dụng module AutoTokenizer chuyên biệt. Công cụ này áp dụng kỹ thuật tách từ phụ (Subword Tokenization - BPE), chia nhỏ các từ hiếm gặp thành những phần nhỏ hơn để giải quyết vấn đề từ vựng ngoài từ điển (Out-Of-Vocabulary). Hơn nữa, vì các mạng nơ-ron đồ thị yêu cầu kích thước đầu vào phải đồng nhất theo lô (batch), mọi chuỗi văn bản đều trải qua kỹ thuật cắt tỉa (Truncation) nếu vượt quá chiều dài cho phép, hoặc được đệm thêm các ký tự rỗng (Padding) cho đến khi đạt đủ độ dài tối đa là max_length = 256 token. Quá trình này biến đổi toàn bộ cơ sở dữ liệu ngôn ngữ thành các ma trận tensor số học cố định, sẵn sàng bơm vào cấu trúc của mạng nơ-ron học sâu.

## 4. Triển khai Hệ thống và Khảo sát Cấu trúc (Implement)

### 4.1. Chiến lược Phân chia Dữ liệu Huấn luyện và Kiểm thử

Nhằm đảm bảo tính khách quan tối đa khi đánh giá hiệu suất của mô hình, tránh việc mô hình vô tình "nhớ" dữ liệu (data leakage) thay vì "hiểu" quy luật phân phối, quy trình phân chia dữ liệu được thực hiện vô cùng cẩn trọng. Thay vì chia tách ngẫu nhiên thông thường, chiến lược phân chia phân tầng (Stratified Splitting) được tích hợp trong thư viện scikit-learn đã được áp dụng. Phương pháp này khóa chặt tỷ lệ phần trăm phân bố của các nhãn cảm xúc và các chủ đề (topic) giữa các tập Training, Validation, và Testing sao cho chúng tương đồng hoàn hảo với phân phối của quần thể gốc. Cụ thể, dữ liệu được chia theo tỷ lệ tiêu chuẩn: phần lớn dữ liệu dùng để tối ưu hóa trọng số (Training Set), một phần nhỏ dùng để theo dõi hiện tượng quá khớp trong quá trình hội tụ (Validation Set), và một phần hoàn toàn độc lập được cách ly nghiêm ngặt để kiểm chứng ở giai đoạn cuối cùng (Testing Set).

### 4.2. Khảo sát Đa Cấu trúc Mô hình Huấn luyện

Sự thành công của một dự án Trí tuệ Nhân tạo không đến từ việc chọn ngẫu nhiên một mô hình phức tạp nhất, mà thông qua quá trình huấn luyện và đối chuẩn (benchmark) trên diện rộng, trải dài từ các khái niệm tuyến tính cơ bản đến các mạng nơ-ron sâu thẳm. Trong bài báo cáo này, chúng tôi đã triển khai một phổ thử nghiệm bao gồm 8 kiến trúc thuật toán chuyên biệt, đại diện cho các trường phái học máy khác nhau:

1. **Nhóm Mô hình Tuyến tính và Thống kê Cổ điển (Baseline Machine Learning)**:
   - Logistic Regression: Hoạt động như một đường cơ sở (baseline) thiết yếu. Nó xây dựng một phương trình tuyến tính dựa trên các đặc trưng rời rạc để ước tính xác suất của từng lớp cảm xúc.
   - Support Vector Machine (SVM): Áp dụng thuật toán tối đa hóa biên (margin maximization) để tìm ra một siêu phẳng (hyperplane) phân tách các cụm từ vựng đa chiều một cách tối ưu nhất. Mặc dù là một mô hình cổ điển, SVM với kernel tuyến tính luôn nổi tiếng về sự mạnh mẽ trong không gian phân loại văn bản đa chiều.

2. **Nhóm Mô hình Học Quần thể (Ensemble Learning)**:
   - Robofort (Random Forest): Như đã phân tích, thuật toán này cấu trúc một "khu rừng" của hàng trăm cây quyết định độc lập. Mỗi cây sẽ bỏ phiếu cho một nhãn cảm xúc, và quyết định cuối cùng dựa trên nguyên tắc số đông (majority voting). Đây là pháo đài vững chắc nhất của họ thuật toán ML truyền thống nhờ khả năng kháng nhiễu và xử lý phi tuyến tốt.
   - Stacked ML Models (Ensemble): Một mô hình xếp chồng meta-learning, huấn luyện một thuật toán cấp cao hơn (meta-classifier) để học cách kết hợp tối ưu các dự đoán từ Logistic Regression, SVM và Random Forest, nhằm khai thác sức mạnh tổng hợp của cả ba phương pháp.

3. **Nhóm Mô hình Học sâu Dữ liệu Chuỗi (Recurrent Deep Learning)**:
   - Fully Connected Layers (Dense): Mô hình mạng nơ-ron tiến thẳng (Feedforward Neural Networks) cơ bản nhất, nhận đầu vào là các véc-tơ nhúng (embeddings) và đi qua các lớp nơ-ron dày đặc, đi kèm với các kỹ thuật chính quy hóa như Dropout (thường ở mức 0.3 - 0.5) để tránh hiện tượng học vẹt.
   - Gated Recurrent Unit (GRU): Để khắc phục việc ML bỏ qua trình tự không gian của câu, GRU được giới thiệu với các cổng (gates) cập nhật và thiết lập lại, cho phép mạng nơ-ron duy trì một bộ nhớ trạng thái ngắn hạn về những từ xuất hiện trước đó trong đoạn văn, từ đó nhận diện được sự thay đổi cảm xúc.
   - Bidirectional LSTM (Bi-LSTM): Phiên bản nâng cấp mạnh mẽ của mạng bộ nhớ ngắn-dài (Long Short-Term Memory). Thay vì chỉ đọc văn bản theo một chiều, Bi-LSTM quét qua câu phản hồi theo cả hai hướng (từ trái sang phải và từ phải sang trái). Kiến trúc này giúp mỗi từ trong câu đều được biểu diễn bởi toàn bộ bối cảnh của các từ xung quanh nó, giải quyết triệt để các cấu trúc ngữ pháp phức tạp.

4. **Nhóm Mô hình Chú ý Tiên tiến (SOTA Transformer)**:
   - PhoBERT: Là điểm nhấn công nghệ và trung tâm của toàn bộ dự án. Khởi nguồn từ kiến trúc RoBERTa, PhoBERT không phân tích câu một cách tuần tự từ đầu đến cuối như RNN, mà xử lý toàn bộ đoạn văn bản cùng một lúc thông qua cơ chế Tự chú ý Đa đầu (Multi-Head Self-Attention). Cơ chế này tạo ra một ma trận trọng số, tính toán trực tiếp sự liên quan của mỗi từ với tất cả các từ khác trong câu, bất kể khoảng cách vật lý của chúng xa đến đâu. Mô hình vinai/phobert-base được khởi tạo cùng hàng chục triệu tham số đã hấp thụ tri thức ngôn ngữ từ hàng chục Gigabyte văn bản tiếng Việt. Quá trình triển khai tinh chỉnh (Fine-tuning) chỉ diễn ra ở các lớp kết nối đầy đủ (Dense Layers) cuối cùng, sử dụng hàm kích hoạt Softmax để xuất ra phân phối xác suất dự đoán cho 3 nhãn phân lớp.

## 5. Đánh giá Khả năng Suy luận và Phân tích Trực quan

### 5.1. Định nghĩa Hệ thống Thang đo (Evaluation Metrics)

Để thẩm định chính xác năng lực phán đoán của một hệ thống trí tuệ nhân tạo, sự phụ thuộc vào một chỉ số đơn lẻ là không đủ. Hệ thống đánh giá dựa vào cấu trúc Ma trận Nhầm lẫn (Confusion Matrix), định nghĩa 4 giá trị cơ sở: True Positives (TP - dự đoán đúng mẫu dương), True Negatives (TN - dự đoán đúng mẫu âm), False Positives (FP - nhận dạng sai mẫu âm thành dương), và False Negatives (FN - bỏ sót mẫu dương). Dựa trên nền tảng này, 4 thang đo vĩ mô được tính toán:

- **Accuracy (Độ chính xác toàn cục)**: Phản ánh tỷ lệ phần trăm các câu phản hồi được mô hình phân loại hoàn toàn trùng khớp với nhãn thực tế do con người đánh giá trên tổng số mẫu kiểm thử.  
  $$ Accuracy = \frac{TP + TN}{TP + TN + FP + FN} $$

- **Precision (Độ chuẩn xác)**: Trong số tất cả các phản hồi mà mô hình dán nhãn là "Tiêu cực", có bao nhiêu phần trăm thực sự là tiêu cực? Chỉ số này đo lường mức độ "chắc chắn" của mô hình, giúp ngăn ngừa báo động giả.  
  $$ Precision = \frac{TP}{TP + FP} $$

- **Recall (Độ phủ/Độ nhạy)**: Trong toàn bộ các phản hồi thực sự mang tính "Tiêu cực" tồn tại trong dữ liệu, mô hình đã "tìm thấy" và phân loại đúng được bao nhiêu phần trăm? Thang đo này tối quan trọng khi nhà trường không muốn bỏ sót bất kỳ một phàn nàn nào của sinh viên.  
  $$ Recall = \frac{TP}{TP + FN} $$

- **F1-Score (Trung bình điều hòa)**: Một số liệu cân bằng tuyệt đối kết hợp cả Precision và Recall. Trong bối cảnh tập dữ liệu UIT-VSFC có sự phân phối bất cân xứng nặng nề (nhãn Neutral quá ít), Accuracy có thể gây ảo giác về hiệu năng. F1-Score (đặc biệt là dạng macro-F1 hoặc weighted-F1) là thước đo khắc nghiệt nhất, chứng minh mô hình thực sự hiểu được các lớp thiểu số.  
  $$ F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall} $$

### 5.2. Phân tích Hiệu năng Bảng Tổng hợp và Best Model Robofort vs PhoBERT

Kết quả trích xuất từ tập kiểm thử độc lập cho thấy bức tranh năng lực đa dạng của toàn bộ hệ thống mô hình được thiết lập.

| Tên Thuật toán (Architecture)          | Độ chính xác (Accuracy) | F1-Score | Độ chuẩn xác (Precision) | Độ phủ (Recall) |
|----------------------------------------|--------------------------|----------|---------------------------|-----------------|
| PhoBERT (SOTA Transformer)            | 96.70%                  | 96.69%  | 96.73%                   | 96.70%         |
| Random Forest (Robofort)              | 93.67%                  | 93.62%  | 93.70%                   | 93.67%         |
| Gated Recurrent Unit (GRU)            | 93.35%                  | 93.33%  | 93.35%                   | 93.35%         |
| Bidirectional LSTM (Bi-LSTM)          | 92.14%                  | 92.08%  | 92.32%                   | 92.14%         |
| Stacked ML Models (Ensemble)          | 90.58%                  | 90.57%  | 90.68%                   | 90.58%         |
| Support Vector Machine (SVM)          | 90.58%                  | 90.55%  | 90.80%                   | 90.58%         |
| Fully Connected Layers (Dense)        | 90.05%                  | 90.00%  | 90.12%                   | 90.05%         |
| Logistic Regression (Baseline)        | 89.70%                  | 89.69%  | 90.01%                   | 89.70%         |

**Bảng 1:** Bảng tham chiếu hiệu năng toàn diện của các mô hình phân loại trên tập dữ liệu kiểm chứng UIT-VSFC (Trích dẫn dựa trên log dữ liệu huấn luyện mã nguồn).

Nhìn vào hệ quy chiếu các phương pháp học máy cổ điển, mô hình Robofort (Random Forest) nổi lên như một điểm sáng rực rỡ nhất khi dẫn đầu với Accuracy đạt 93.67% và F1-Score đạt 93.62%. Trong một không gian đặc trưng tuyến tính có mật độ nhiễu cao của ngôn ngữ mạng, việc Robofort vượt qua cột mốc 93% là một minh chứng xuất sắc cho khả năng tinh gọn dữ liệu và phân tách không gian đa chiều của các thuật toán phân nhánh dựa trên Entropy. Nó dễ dàng đè bẹp các phương pháp phân loại dựa trên lề (margin) như SVM (90.58%) hay phương pháp tiếp cận mạng nơ-ron dày đặc truyền thống (Dense - 90.05%). Đánh giá về Best Robofort cho thấy, mô hình này là sự lựa chọn ưu tiên hàng đầu nếu hệ thống yêu cầu tốc độ phản hồi tính bằng mili-giây và chạy trên các phần cứng nghèo nàn tài nguyên, nhờ vào cấu trúc biểu diễn If-Else nội tại của nó.

Tuy nhiên, dù xuất sắc đến mấy, các mô hình dựa trên Bag-of-Words như Robofort đã chạm phải bức tường giới hạn ngữ nghĩa (semantic ceiling). Khi phải đối mặt với các hiện tượng ngôn ngữ đảo ngữ, mỉa mai ngầm, sự thống trị tuyệt đối đã hoàn toàn thuộc về PhoBERT.

Ưu điểm làm nên sức mạnh hủy diệt của PhoBERT chính là cơ chế Attention. Nhờ việc được huấn luyện trước trên 20GB văn bản tiếng Việt chuẩn, PhoBERT hiểu được mối quan hệ logic giữa các danh từ và tính từ, giúp nó không bị đánh lừa bởi những từ vựng nhạy cảm nằm rải rác trong câu. Với Accuracy lên tới 96.70% và F1-Score gần tương đương 96.69%, PhoBERT chứng tỏ nó không chỉ học tốt mà còn giải quyết được sự bất cân bằng của nhãn Trung lập một cách kiệt xuất. Độ chuẩn xác (Precision) cao chót vót ở mức 96.73% đảm bảo rằng những phản hồi bị gán nhãn là Tiêu cực thực sự là những phàn nàn cần giải quyết, giúp đội ngũ quản lý trường học tiết kiệm tối đa thời gian xác minh lại.

### 5.3. Phân tích Biểu đồ Lịch sử Huấn luyện (Training History) của PhoBERT

Việc quan sát biểu đồ hàm mất mát (Loss) và độ chính xác (Accuracy) qua từng kỷ nguyên huấn luyện (Epochs) là cơ sở khoa học để đánh giá tính ổn định của mạng nơ-ron. Dựa trên nhật ký xuất log từ mã nguồn thuật toán, quá trình hội tụ của PhoBERT diễn ra mạnh mẽ và mượt mà.

![Hình 2: Biểu đồ trực quan hóa diễn biến Hàm Mất mát (Loss) và Độ Chính xác (Accuracy) của PhoBERT qua 3 chu kỳ huấn luyện]

**Hình 2:** Biểu đồ trực quan hóa diễn biến Hàm Mất mát (Loss) và Độ Chính xác (Accuracy) của PhoBERT qua 3 chu kỳ huấn luyện.

Phân tích sâu vào các chỉ số quá trình diễn ra như sau:

- **Kỷ nguyên 1 (Epoch 1)**: Hệ thống bắt đầu bước vào quá trình tinh chỉnh (Fine-tuning). Hàm mất mát của tập huấn luyện (Train Loss) mở màn ở mức 0.3450, kéo theo độ chính xác trên tập Train đạt 87.02%. Thế nhưng, một điều kỳ diệu của kỹ thuật Transfer Learning xuất hiện: ngay từ chu kỳ đầu tiên, độ chính xác trên tập Validation (Val_Accuracy) đã vọt lên mức 95.72% đi kèm với Val_Loss cực thấp 0.1434. Hiện tượng này chứng minh rằng các trọng số tiền huấn luyện từ tập dữ liệu Wikipedia của PhoBERT đã tự thân nó mang sẵn một bộ biểu diễn ngữ nghĩa tiếng Việt vô cùng hoàn hảo, chỉ cần một vài vòng lặp nhẹ là đã có thể thích ứng ngay với ngôn ngữ của sinh viên.

- **Kỷ nguyên 2 (Epoch 2)**: Mô hình bắt đầu làm chủ các sắc thái của tập dữ liệu UIT-VSFC. Lỗi trên tập Train tụt dốc không phanh xuống 0.1455, trong khi Train Accuracy vươn lên 96.17%. Val_Accuracy đạt đỉnh điểm hội tụ tại 96.94% với Val_Loss chạm đáy 0.1169. Đây là thời điểm ranh giới siêu phẳng đa chiều của PhoBERT được định hình rõ rệt nhất.

- **Kỷ nguyên 3 (Epoch 3)**: Mạng nơ-ron tiệm cận trạng thái bão hòa (saturation). Train Loss giảm xuống ngưỡng cực tiểu 0.0865, đẩy Train Accuracy lên 97.59%. Trong khi đó, Val_Accuracy duy trì sự ổn định ở mốc 96.70%. Việc đường cong Validation đi song song sát nút với đường cong Training mà không có hiện tượng vểnh lên (divergence) khẳng định chắc chắn rằng mô hình không hề bị quá khớp (Overfitting) với dữ liệu mẫu. Một sự hội tụ hoàn mĩ.

## 6. Đường ống Triển khai và Môi trường Giả lập (Deploy & Demo)

Mọi nghiên cứu về Trí tuệ Nhân tạo sẽ chỉ là các tham số vô hồn nếu không được đưa vào ứng dụng thực tế. Nhằm chuyển đổi kiến trúc mạng phức tạp thành một công cụ phân tích thân thiện cho giảng viên và nhà trường, dự án đã xây dựng một luồng triển khai (Deployment Pipeline) chuyên nghiệp trên nền tảng điện toán đám mây.

### 6.1. Đóng gói Mô hình và Khởi tạo Môi trường

Sau khi hoàn thành Epoch cuối cùng, trọng số khổng lồ của mô hình Học sâu (keras_model) cùng với bộ tự điển chuyên dụng (tokenizer) của PhoBERT được đóng gói cẩn mật thành một tập tin nén vĩ mô phobert_production_bundle.zip. Cơ chế nén này đóng vai trò quan trọng trong việc tiết kiệm không gian lưu trữ và đảm bảo băng thông mạng truyền tải khi hệ thống khởi động trên nền tảng đám mây.

Để giải quyết vấn đề tương thích khét tiếng của hệ sinh thái phần mềm Học sâu, kiến trúc triển khai được thiết lập với cờ môi trường cưỡng bức os.environ = "1". Việc sử dụng backend Keras kế thừa này nhằm bảo tồn sự toàn vẹn của cấu trúc Transformer khi tải lên qua hàm keras.models.load_model(keras_path, compile=False), giúp ngăn chặn các lỗi tràn bộ nhớ hoặc rò rỉ biểu đồ tính toán không đáng có khi phục vụ yêu cầu thời gian thực.

### 6.2. Kiến trúc Giao diện Đa tầng với Gradio

Thay vì yêu cầu người dùng cuối thao tác thông qua các giao diện lập trình ứng dụng (API) khô khan, hệ thống tích hợp trực tiếp với thư viện Gradio để sinh ra một giao diện Web trực quan (Intuitive Web UI). Thiết kế giao diện (UI Design) được phân tách khoa học thành hai phân hệ (Modules) xử lý độc lập để phục vụ các quy trình nghiệp vụ khác nhau:

**Phân hệ Phân tích Đơn lẻ (Single Inference)**:  
Được thiết kế phục vụ cho nhu cầu kiểm tra nhanh. Giảng viên nhập một đoạn nhận xét ngẫu nhiên vào hộp thoại. Hệ thống chuyển tiếp chuỗi văn bản vào luồng làm sạch (xóa emojis, chuẩn hóa tiếng Việt) theo đúng chuẩn của quy trình đào tạo, sau đó trích xuất tensor độ dài cố định qua Tokenizer. Chỉ trong vòng vài mili-giây, luồng dữ liệu truyền qua các nơ-ron mạng và dội ngược lại một phân phối Softmax. Giao diện xuất ra một Biểu đồ dải ngang (Horizontal Bar Chart) cực kỳ trực quan, trình bày chi tiết phần trăm khả năng rơi vào các khoảng cảm xúc (Tiêu cực, Trung lập, Tích cực), cung cấp diễn giải minh bạch cho mức độ niềm tin của hệ thống.

**Phân hệ Phân tích Lô Doanh nghiệp (Batch Inference)**:  
Đây là trái tim nghiệp vụ của hệ thống, cho phép cán bộ quản lý tải lên toàn bộ kết quả khảo sát dưới định dạng tệp tin bảng tính (.csv, .xlsx). Hệ thống ứng dụng một thuật toán nhận diện thông minh, tự động lặp qua metadata của bảng tính để định vị chính xác vị trí cột chứa phản hồi văn bản dựa trên các từ khóa logic như 'comment', 'text', 'nhận xét' hoặc 'nội dung'.

![Hình 3: Giao diện trực quan hóa kết quả phân tích theo lô bằng Biểu đồ hình tròn Matplotlib]

**Hình 3:** Giao diện trực quan hóa kết quả phân tích theo lô bằng Biểu đồ hình tròn Matplotlib.

Sau khi chạy tiến trình dự báo tuần tự qua mô hình PhoBERT cho toàn bộ tệp dữ liệu, thuật toán tổng hợp thư viện Matplotlib vẽ ra một Biểu đồ hình tròn (Pie Chart) cao cấp. Biểu đồ được thiết kế với giao diện vô hình (transparent background) hòa hợp tinh tế với chế độ tối (Dark Mode) của hệ thống máy chủ. Các mảng cảm xúc được ánh xạ bởi bộ mã màu tiêu chuẩn: Xanh lá (#2ecc71) cho Tích cực, Đỏ (#e74c3c) cho Tiêu cực và Xám (#95a5a6) cho Trung lập. Đặc biệt, mảng cảm xúc có tỷ trọng lớn nhất được tạo hiệu ứng tách rời (Explode 0.08) đổ bóng vật lý (Shadow), kết hợp phông chữ trắng in đậm và chú giải (Legend) nổi, mang lại trải nghiệm báo cáo thống kê chuyên nghiệp như các phần mềm Business Intelligence hàng đầu.

### 6.3. Triển khai Lên Đám mây (Hugging Face Spaces) và Khả năng Thực thi Cục bộ

Toàn bộ mã nguồn ứng dụng (app.py), cùng với tệp tin requirements.txt định nghĩa chặt chẽ phiên bản của các thư viện (như tensorflow, tf-keras, underthesea, transformers), được triển khai tích hợp liên tục (CI/CD) lên nền tảng đám mây Hugging Face Spaces. Tại đây, hệ thống sở hữu cơ chế mồi khởi động (Bootstrap mechanism): ngay khi máy chủ đám mây bắt đầu vòng đời, đoạn mã Python sẽ kiểm tra sự tồn tại của thư mục mô hình. Nếu chưa có, hệ thống tự động giải nén tệp .zip và đặt cấu trúc thư mục vào vị trí chính xác để TensorFlow tiến hành tải mô hình lên RAM tĩnh.

Ngoài ra, hệ thống cũng được thiết kế mở hoàn toàn, cung cấp tài liệu kỹ thuật để bất kỳ nhà nghiên cứu nào cũng có thể sao chép kho chứa (git clone) và thiết lập môi trường giả lập máy chủ Gradio cục bộ (Localhost). Sự kết hợp giữa năng lực tính toán siêu hạng của mạng nơ-ron Transformer và một kiến trúc phần mềm dễ tiếp cận đã biến dự án này thành một khối tài sản giá trị, mở ra tiền lệ mới cho công tác đánh giá hệ thống đào tạo bằng trí tuệ nhân tạo tại Việt Nam.
