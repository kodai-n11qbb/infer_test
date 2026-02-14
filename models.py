#!/usr/bin/env python3
"""
モデル推論モジュール - 改造しやすい構造
車線検出、物体検出、意思決定を独立したクラスで実装
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import time

@dataclass
class DetectionResult:
    """検出結果の標準データ形式"""
    class_name: str
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    distance: float
    additional_info: Dict[str, Any] = None

@dataclass
class LaneResult:
    """車線検出結果の標準データ形式"""
    left_lane: Optional[List[int]] = None  # [x1, y1, x2, y2]
    right_lane: Optional[List[int]] = None
    lane_center: Optional[int] = None
    deviation: Optional[int] = None

@dataclass
class DrivingCommand:
    """運転指示の標準データ形式"""
    action: str
    confidence: float
    urgency: int  # 1-10
    reason: str
    metadata: Dict[str, Any] = None

class BaseModel:
    """全モデルの基底クラス - 共通機能を提供"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.frame_count = 0
        self.processing_time = 0.0
    
    def update_stats(self, start_time: float):
        """処理時間統計を更新"""
        self.processing_time = time.time() - start_time
        self.frame_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
        avg_time = self.processing_time / self.frame_count if self.frame_count > 0 else 0
        return {
            'frames_processed': self.frame_count,
            'avg_processing_time': avg_time,
            'fps': 1.0 / avg_time if avg_time > 0 else 0
        }

class LaneDetectionModel(BaseModel):
    """車線検出モデル - ここを改造して新しいアルゴリズムを試せる"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        # 設定パラメータ
        self.canny_low = self.config.get('canny_low', 50)
        self.canny_high = self.config.get('canny_high', 150)
        self.hough_threshold = self.config.get('hough_threshold', 50)
        self.min_line_length = self.config.get('min_line_length', 40)
        self.max_line_gap = self.config.get('max_line_gap', 20)
        self.slope_threshold = self.config.get('slope_threshold', 0.5)
    
    def detect(self, frame: np.ndarray) -> LaneResult:
        """
        車線を検出 - メイン処理
        ここに新しい車線検出アルゴリズムを実装できる
        """
        start_time = time.time()
        
        try:
            # 前処理
            processed = self._preprocess(frame)
            
            # 線分検出
            lines = self._detect_lines(processed)
            
            # 車線解析
            left_lane, right_lane = self._analyze_lines(lines, frame.shape)
            
            # 中心線計算
            lane_result = self._calculate_lane_info(left_lane, right_lane, frame.shape)
            
            self.update_stats(start_time)
            return lane_result
            
        except Exception as e:
            print(f"車線検出エラー: {e}")
            return LaneResult()
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """画像前処理 - ここで新しい前処理を追加できる"""
        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ガウシアンブラー
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Cannyエッジ検出
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # 関心領域マスク
        height, width = edges.shape
        roi_vertices = np.array([
            [(0, height), (width * 0.45, height * 0.6), 
             (width * 0.55, height * 0.6), (width, height)]
        ], dtype=np.int32)
        
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        return masked_edges
    
    def _detect_lines(self, processed: np.ndarray) -> List[List[int]]:
        """線分検出 - ハフ変換"""
        lines = cv2.HoughLinesP(processed, 1, np.pi/180, self.hough_threshold, 
                               minLineLength=self.min_line_length, 
                               maxLineGap=self.max_line_gap)
        
        if lines is None:
            return []
        
        return [line[0].tolist() for line in lines]
    
    def _analyze_lines(self, lines: List[List[int]], shape: Tuple[int, int, int]) -> Tuple[Optional[List[int]], Optional[List[int]]]:
        """線分を左右車線に分類"""
        height, width = shape[:2]
        center_x = width // 2
        
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            if x2 - x1 == 0:
                continue
                
            slope = (y2 - y1) / (x2 - x1)
            
            if slope < -self.slope_threshold:  # 左車線
                left_lines.append(line)
            elif slope > self.slope_threshold:  # 右車線
                right_lines.append(line)
        
        left_lane = self._average_lines(left_lines) if left_lines else None
        right_lane = self._average_lines(right_lines) if right_lines else None
        
        return left_lane, right_lane
    
    def _average_lines(self, lines: List[List[int]]) -> List[int]:
        """複数線分を平均化"""
        if not lines:
            return []
        
        slopes = []
        intercepts = []
        
        for x1, y1, x2, y2 in lines:
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            slopes.append(slope)
            intercepts.append(intercept)
        
        if not slopes:
            return []
        
        avg_slope = np.mean(slopes)
        avg_intercept = np.mean(intercepts)
        
        height = 480  # 仮の高さ
        y1 = height
        y2 = int(height * 0.6)
        x1 = int((y1 - avg_intercept) / (avg_slope + 1e-6))
        x2 = int((y2 - avg_intercept) / (avg_slope + 1e-6))
        
        return [x1, y1, x2, y2]
    
    def _calculate_lane_info(self, left_lane: Optional[List[int]], 
                           right_lane: Optional[List[int]], 
                           shape: Tuple[int, int, int]) -> LaneResult:
        """車線情報を計算"""
        height, width = shape[:2]
        
        if left_lane and right_lane:
            lane_center = (left_lane[0] + right_lane[0]) // 2
            frame_center = width // 2
            deviation = lane_center - frame_center
        else:
            lane_center = None
            deviation = None
        
        return LaneResult(
            left_lane=left_lane,
            right_lane=right_lane,
            lane_center=lane_center,
            deviation=deviation
        )

class ObjectDetectionModel(BaseModel):
    """物体検出モデル - ここを改造して新しいモデルを試せる"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model_type = self.config.get('model_type', 'contour')  # 'contour', 'yolo', 'custom'
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        
        # モデル初期化
        self._initialize_model()
    
    def _initialize_model(self):
        """モデルを初期化 - ここで新しいモデルをロードできる"""
        if self.model_type == 'yolo':
            self._load_yolo_model()
        elif self.model_type == 'custom':
            self._load_custom_model()
        else:
            self._setup_contour_detection()
    
    def _load_yolo_model(self):
        """YOLOモデルをロード"""
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO('yolov8n.pt')
            self.target_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person']
        except ImportError:
            print("YOLOモデルをロードできません。コンター検出に切り替えます")
            self.model_type = 'contour'
            self._setup_contour_detection()
    
    def _load_custom_model(self):
        """カスタムモデルをロード - ここに新しいモデルを実装"""
        # 例: TensorFlow Liteモデル、ONNXモデルなど
        print("カスタムモデルロード機能 - ここに実装を追加")
        self.model_type = 'contour'
        self._setup_contour_detection()
    
    def _setup_contour_detection(self):
        """輪郭検出のセットアップ"""
        self.yolo_model = None
    
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        物体を検出 - メイン処理
        ここに新しい物体検出アルゴリズムを実装できる
        """
        start_time = time.time()
        
        try:
            if self.model_type == 'yolo' and self.yolo_model:
                results = self._detect_with_yolo(frame)
            else:
                results = self._detect_with_contours(frame)
            
            self.update_stats(start_time)
            return results
            
        except Exception as e:
            print(f"物体検出エラー: {e}")
            return []
    
    def _detect_with_yolo(self, frame: np.ndarray) -> List[DetectionResult]:
        """YOLOで物体検出"""
        results = self.yolo_model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.yolo_model.names[class_id]
                    
                    if class_name in self.target_classes and confidence > self.confidence_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        distance = self._estimate_distance(x2 - x1, y2 - y1, class_name)
                        
                        detections.append(DetectionResult(
                            class_name=class_name,
                            bbox=[int(x1), int(y1), int(x2), int(y2)],
                            confidence=confidence,
                            distance=distance,
                            additional_info={'model': 'yolo'}
                        ))
        
        return detections
    
    def _detect_with_contours(self, frame: np.ndarray) -> List[DetectionResult]:
        """輪郭検出で物体検出"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 赤と青の色範囲
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask = cv2.bitwise_or(mask_red, mask_blue)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            distance = max(2.0, min(50.0, 10000 / (w * h)))
            
            center_x = x + w // 2
            center_y = y + h // 2
            pixel_color = hsv[center_y, center_x]
            
            class_name = "car" if pixel_color[0] < 10 or pixel_color[0] > 170 else "truck"
            confidence = min(0.9, area / 10000)
            
            detections.append(DetectionResult(
                class_name=class_name,
                bbox=[x, y, x + w, y + h],
                confidence=confidence,
                distance=distance,
                additional_info={'model': 'contour', 'area': area}
            ))
        
        return detections
    
    def _estimate_distance(self, width: float, height: float, class_name: str) -> float:
        """距離を推定 - ここで新しい距離推定アルゴリズムを実装できる"""
        reference_sizes = {
            'car': 100, 'truck': 150, 'bus': 180,
            'motorcycle': 60, 'bicycle': 50, 'person': 40
        }
        
        ref_size = reference_sizes.get(class_name, 100)
        estimated_distance = ref_size / max(width, height) * 10
        return max(1.0, min(estimated_distance, 100.0))

class DecisionMakingModel(BaseModel):
    """意思決定モデル - ここを改造して新しいアルゴリズムを試せる"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.emergency_distance = self.config.get('emergency_distance', 5.0)
        self.safe_distance = self.config.get('safe_distance', 15.0)
        self.deviation_threshold = self.config.get('deviation_threshold', 50)
    
    def decide(self, lane_result: LaneResult, objects: List[DetectionResult]) -> DrivingCommand:
        """
        運転指示を決定 - メイン処理
        ここに新しい意思決定アルゴリズムを実装できる
        """
        start_time = time.time()
        
        try:
            # 緊急事態チェック
            emergency = self._check_emergency(objects)
            if emergency:
                self.update_stats(start_time)
                return emergency
            
            # 車線逸脱チェック
            lane_command = self._check_lane_deviation(lane_result)
            if lane_command:
                self.update_stats(start_time)
                return lane_command
            
            # 前方障害物チェック
            obstacle_command = self._check_obstacles(objects)
            if obstacle_command:
                self.update_stats(start_time)
                return obstacle_command
            
            # デフォルト指示
            default_command = DrivingCommand(
                action="STRAIGHT",
                confidence=0.8,
                urgency=1,
                reason="正常な直進状態",
                metadata={'strategy': 'default'}
            )
            
            self.update_stats(start_time)
            return default_command
            
        except Exception as e:
            print(f"意思決定エラー: {e}")
            return DrivingCommand(
                action="STRAIGHT",
                confidence=0.5,
                urgency=2,
                reason="エラー時の安全動作",
                metadata={'error': str(e)}
            )
    
    def _check_emergency(self, objects: List[DetectionResult]) -> Optional[DrivingCommand]:
        """緊急事態をチェック"""
        for obj in objects:
            if obj.distance < self.emergency_distance:
                return DrivingCommand(
                    action="BRAKE",
                    confidence=0.95,
                    urgency=10,
                    reason=f"緊急：{obj.class_name}が{obj.distance:.1f}mに接近",
                    metadata={'emergency_object': obj.class_name, 'distance': obj.distance}
                )
        return None
    
    def _check_lane_deviation(self, lane_result: LaneResult) -> Optional[DrivingCommand]:
        """車線逸脱をチェック"""
        if lane_result.deviation and abs(lane_result.deviation) > self.deviation_threshold:
            direction = "TURN_RIGHT" if lane_result.deviation > 0 else "TURN_LEFT"
            return DrivingCommand(
                action=direction,
                confidence=0.8,
                urgency=5,
                reason=f"車線逸脱：偏差{lane_result.deviation}px",
                metadata={'deviation': lane_result.deviation}
            )
        return None
    
    def _check_obstacles(self, objects: List[DetectionResult]) -> Optional[DrivingCommand]:
        """前方障害物をチェック"""
        frame_center = 320  # 640pxの中心
        front_objects = []
        
        for obj in objects:
            obj_center_x = (obj.bbox[0] + obj.bbox[2]) // 2
            if abs(obj_center_x - frame_center) < 100:  # 前方領域
                front_objects.append(obj)
        
        if front_objects:
            closest = min(front_objects, key=lambda x: x.distance)
            if closest.distance < self.safe_distance:
                return DrivingCommand(
                    action="SLOW_DOWN",
                    confidence=0.7,
                    urgency=4,
                    reason=f"前方{closest.class_name}に減速：{closest.distance:.1f}m",
                    metadata={'front_object': closest.class_name, 'distance': closest.distance}
                )
        
        return None
