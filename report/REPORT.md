# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 6/11/2026

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** 3.13.16/2.11.0+cu130/8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` 

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1.  {
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "image_classification",
    "taxonomy_name": "ImageNet-1K",
    "model_file": "yolo11n-cls.pt",
    "model_sha256": "c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7",
    "ultralytics_version": "8.4.145",
    "rank": 1,
    "class_id": 468,
    "class_name": "cab",
    "score": 0.510915
  },
2. Record này nói về một vật thể có nhãn là "cab", id = 468, confidence score=0.510915
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
		Class list đi kèm với taxonomy/checkpoint
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
		Vì để phân biệt giữa id mô hình dùng, tên class đọc được, và vị trí của nhãn đó ở trong danh sách nhãn nào?
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
		Guideline cần mô tả chi tiết, rõ ràng từng nhãn
6. Vì sao không được lấy `score` của model làm ground truth?
		Vì score của model chỉ thể hiện độ chắc chắn của model khi dự đoán, không mô tả sự thật khách quan của data

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
		{
    "sample_id": "kitchen",
    "coco_image_id": 397133,
    "image_width": 640,
    "image_height": 427,
    "task": "object_detection",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n.pt",
    "model_sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "class_id": 0,
    "class_name": "person",
    "score": 0.912625,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      385.33,
      69.24,
      498.92,
      348.92
    ],
    "bbox_width": 113.58,
    "bbox_height": 279.68
  },
1. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
		góc trên-trái ở [385.33, 498.92], góc dưới-trái [69.24, 348.92]. Chiều rộng 113.58, chiều dài 279.68
2. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
		giữa 2 ngưỡng 0.20 và 0.60, có 39 ngưỡng từ 0.21 đến 0.59
3. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
		Khi hạ ngưỡng thì reviewer cần check kỹ dự đoán hơn, do hạ ngưỡng thì sẽ có nhiều bounding box hơn, mô hình sẽ in ra nhiều dự đoán sai hơn
4. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
		Bounding box phải sát đối tượng nhất có thể, nhưng không được cắt mất đối tượng
5. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
		Ảnh kitchen có một bàn tay người (score=0.61) bị khuất vẫn được đánh nhãn person. Trường hợp này cần đưa lên hỏi Lab Coach, vì guideline khó có thể cover cụ thể từng trường hợp
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
		{
    "sample_id": "kitchen",
    "coco_image_id": 397133,
    "image_width": 640,
    "image_height": 427,
    "task": "instance_segmentation",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n-seg.pt",
    "model_sha256": "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "instance_id": "kitchen-001",
    "class_id": 0,
    "class_name": "person",
    "score": 0.899318,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      385.45,
      66.44,
      498.02,
      348.58
    ],
    "polygon_point_count": 348,
    "polygon_xy": [
      [
        446.0,
        70.0
      ],
      [
        445.0,
        71.0
      ],
      [
        444.0,
        71.0
      ],
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
		so với box, polygon cho phép biết về hình dạng cụ thể của vật thể
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
		instance_id là mã định danh giữa các đối tượng có cùng class
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
		Mask phải vẽ theo đường biên của đối tượng, không vẽ lên vùng không thuộc đối tượng
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
		Vật thể owen và person(score 0.43) có biên chạm nhau. Trường hợp này có thể dùng guideline ưu tiên vẽ vật ở phía trước, gần camera hơn đè lên trên vật phía sau.
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ                | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì?                    | Reviewer xem gì?                                            |
| --------------------- | --------------------------------------------- | -------------------------------------------------- | ------------------------------------ | ---------------------------------------------------------- |
| Phân loại ảnh         | cab                                           |                                                    |                                                                                                    |
| Phát hiện vật thể     | person                                        | Vật thể person bị gán thừa nhãn                    | Kiểm tra guideline, không thêm class  | Kiểm tra tính hợp lệ của nhãn, kiểm tra bbox có khớp không
| Instance segmentation | person, owen                                  | Hai vật thể person và owen có biên chồng nhau      | vẽ polygon cho từng vật thể          | Kiểm tra tính đúng đắn của nhãn                                 |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
		Không ghi thông tin cá nhân trong báo cáo/output
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
		Nếu gặp ảnh/data không đúng phạm vi, sẽ thực hiện báo cho Lab coach

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
