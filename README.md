# Tối ưu hóa danh mục đầu tư chứng khoán dựa trên dữ liệu thị trường và dữ liệu tài chính của doanh nghiệp

Mã nguồn và dữ liệu kèm theo luận văn thạc sĩ ngành Khoa học Dữ liệu,
Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM.

Tác giả: Trần Thị Lan Anh (2026).

## Cấu trúc

| Tệp / thư mục | Vai trò |
|---|---|
| `experiment.py` | Nạp dữ liệu, định nghĩa các kịch bản đầu vào và mô hình GRU |
| `stage2_v5.py` | **Bản chạy chính thức**: với mỗi seed, huấn luyện một mô hình gốc duy nhất rồi sinh đồng thời bốn phiên bản danh mục (gốc, đầu vào, lọc ROEA, phân bổ ROEA) trên cùng các ngày lập danh mục |
| `run_experiments.py` | Chạy riêng các thí nghiệm tầng đầu vào |
| `fetch_financial_data.py`, `ThuThap_Ratio_MoRong.ipynb` | Mã thu thập dữ liệu giá và chỉ số tài chính qua thư viện `vnstock` |
| `prepared_2018_2026.csv` | Bộ dữ liệu đã xử lý: 46 cổ phiếu HOSE, 4/2018–3/2026 (1.997 phiên), giá điều chỉnh, khối lượng và 4 chỉ số P/B, P/E, ROAA, ROEA đã áp nguyên tắc point-in-time (trễ 45 ngày theo Thông tư 96/2020/TT-BTC) |
| `ThucNghiem_HopNhat.ipynb` | Notebook điều phối toàn bộ trên Google Colab |
| `results_final_8seeds/` | Kết quả chính thức theo từng seed (42–49), nguồn của các Bảng 4.1–4.7 trong luận văn |

## Môi trường

Python 3, TensorFlow 2.x, scikit-learn, pandas, numpy. Kết quả trong luận văn
được chạy trên GPU T4 của Google Colab.

## Trình tự tái lập

1. Tải toàn bộ thư mục này lên Google Drive.
2. Mở `ThucNghiem_HopNhat.ipynb` bằng Google Colab (chọn runtime GPU), chạy lần
   lượt các ô. Các ô đầu tự kiểm tra sự đầy đủ của tệp; ô chạy chính tự bỏ qua
   các seed đã hoàn thành nên có thể chạy lại an toàn sau gián đoạn.
3. Ô cuối tổng hợp kết quả thành các bảng tương ứng Bảng 4.2–4.6 của luận văn.

Trên cùng loại phần cứng (GPU T4), toàn bộ kết quả tái lập chính xác từng chữ
số nhờ cố định seed (42–49). Trên phần cứng khác, kết quả huấn luyện có thể
lệch nhẹ do khác biệt dấu phẩy động; riêng hai quy tắc lọc và phân bổ là tất
định nên luôn cho cùng kết quả trên cùng danh mục đầu vào.

Lưu ý: notebook có một số ô phân tích bổ sung (gắn nhãn riêng) dùng dữ liệu
không kèm trong gói này; các ô đó không ảnh hưởng đến việc tái lập kết quả
chính và có thể bỏ qua.

## Ghi công

- Khung mô hình kế thừa **DELAFO** (Cao Ky Hieu và cộng sự, 2020, MIT License):
  https://github.com/caokyhieu/DELAFO
- Thiết kế thực nghiệm tiếp nối luận văn của **Phan Thị Thùy An** (2023),
  Trường ĐH KHTN, ĐHQG-HCM.
- Dữ liệu truy xuất qua thư viện mã nguồn mở `vnstock`.

Dữ liệu được cung cấp duy nhất cho mục đích nghiên cứu và kiểm chứng kết quả
luận văn; không phải khuyến nghị đầu tư.
