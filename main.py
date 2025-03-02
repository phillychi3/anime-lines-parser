import os
import cv2
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from paddleocr import PaddleOCR
from pathlib import Path
from ultralytics import YOLO
import torch
from uuid import uuid4


class AnimeVideoAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用設備: {self.device}")
        print("載入 PaddleOCR 模型...")
        self.reader = PaddleOCR(
            use_angle_cls=True, lang="ch", use_gpu=torch.cuda.is_available()
        )
        print("載入情緒分析模型...")
        self.emotion_tokenizer = AutoTokenizer.from_pretrained(
            "Johnson8187/Chinese-Emotion-Small"
        )
        self.emotion_model = AutoModelForSequenceClassification.from_pretrained(
            "Johnson8187/Chinese-Emotion-Small"
        ).to(self.device)

        self.label_mapping = {
            0: "normal",
            1: "concerned",
            2: "happy",
            3: "anger",
            4: "sad",
            5: "question",
            6: "surprise",
            7: "disgusting",
        }

        print("載入 YOLO 模型...")
        self.object_detector = YOLO("yolov8n.pt")
        print("模型載入完成！")

    def extract_subtitle_region(self, frame):
        height = frame.shape[0]
        width = frame.shape[1]

        start_y = int(height * 0.8)
        end_y = height - 30
        start_x = 30
        end_x = width - 30
        frame_with_region = frame.copy()
        cv2.rectangle(
            frame_with_region, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2
        )
        cv2.imshow("字幕辨識區域", frame_with_region)
        cv2.waitKey(1)
        subtitle_region = frame[start_y:end_y, start_x:end_x]
        return subtitle_region

    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)

        _, binary = cv2.threshold(
            contrast, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return binary

    def detect_objects(self, frame):
        results = self.object_detector(frame)
        objects = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.object_detector.names[cls]
                if conf > 0.5:
                    objects.append({"class": class_name, "confidence": conf})

        return objects

    def analyze_emotion(self, text):
        try:
            inputs = self.emotion_tokenizer(
                text, return_tensors="pt", truncation=True, padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.emotion_model(**inputs)

            predicted_class = torch.argmax(outputs.logits).item()
            predicted_emotion = self.label_mapping[predicted_class]

            return predicted_emotion
        except Exception as e:
            print(f"情緒分析錯誤: {str(e)}")
            return "unknown"

    def get_video_info(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()

        return {"fps": fps, "total_frames": total_frames, "duration": duration}

    def process_text(self, text: str):
        return text.strip().replace('"', "").replace("'", "")

    def process_video(self, video_path, result_file_name, output_dir="output"):
        os.makedirs(output_dir, exist_ok=True)
        video_info = self.get_video_info(video_path)

        video = cv2.VideoCapture(video_path)
        frame_count = 0
        results = []
        try:
            pbar = tqdm(total=video_info["total_frames"], desc="處理影片")
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break

                if frame_count % video_info["fps"] == 0:
                    current_second = frame_count / video_info["fps"]
                    subtitle_region = self.extract_subtitle_region(frame)
                    processed_region = self.preprocess_frame(subtitle_region)

                    ocr_result = self.reader.ocr(processed_region, cls=True)
                    current_text = ""

                    if ocr_result:
                        current_text = ""
                        try:
                            if ocr_result[0] is not None:
                                text_results = []
                                for line in ocr_result[0]:
                                    if (
                                        line
                                        and len(line) > 1
                                        and line[1]
                                        and len(line[1]) > 0
                                    ):
                                        text_results.append(line[1][0])
                                current_text = " ".join(text_results)
                        except (IndexError, TypeError) as e:
                            print(f"處理 OCR 結果出錯: {e}")

                    objects = self.detect_objects(frame)
                    if current_text.strip():
                        current_text = self.process_text(current_text)
                        emotion = self.analyze_emotion(current_text)
                        small_id = str(uuid4())[:8]
                        frame_filename = f"{small_id}-{current_second:.1f}s.jpg"
                        results.append(
                            {
                                "frame": frame_count,
                                "time": f"{current_second:.1f}",
                                "text": current_text,
                                "emotion": emotion,
                                "objects": [obj["class"] for obj in objects],
                                "frame_filename": frame_filename,
                            }
                        )
                        cv2.imwrite(
                            Path(output_dir, frame_filename),
                            frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
                        )
                frame_count += 1
                pbar.update(1)
        finally:
            video.release()
            cv2.destroyAllWindows()
            pbar.close()

        if results:
            df = pd.DataFrame(results)
            df.to_csv(
                Path(output_dir, result_file_name + ".csv"),
                index=False,
                encoding="utf-8-sig",
            )
            df.to_json(
                Path(output_dir, result_file_name + ".json"),
                orient="records",
                force_ascii=False,
            )
            print(f"\n分析結果已保存到: {Path(output_dir, result_file_name)}")

        return {
            "total_frames": video_info["total_frames"],
            "fps": video_info["fps"],
            "duration": video_info["duration"],
            "analyzed_frames": len(results),
        }


if __name__ == "__main__":
    output_dir = Path("output")
    analyzer = AnimeVideoAnalyzer()
    video_path = r"C:\Users\whitecloud\Downloads\aniGamerPlus_v24.6_windows_64bit\bangumi\BanG Dream! Ave Mujica\BanG Dream! Ave Mujica[1][720P].mp4"
    csv_name = "anime_subtitles_with_tags"
    results = analyzer.process_video(
        video_path=video_path, result_file_name=csv_name, output_dir=output_dir
    )
    print("處理完成！結果已保存到:", output_dir)
