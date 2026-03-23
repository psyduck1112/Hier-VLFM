# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
# Modifications Copyright (c) 2026 Yikang.

from typing import List

import numpy as np

from .server_wrapper import send_request

try:
    # 尝试导入ultralytics库中的YOLO模型和OpenCV库
    import cv2
    from ultralytics import YOLO

    # 如果导入成功，标记为可用
    ULTRALYTICS_AVAILABLE = True
except ImportError as e:
    # 如果导入失败，打印错误信息并标记为不可用
    print(f"Ultralytics dependencies not available: {e}, will use fallback")
    ULTRALYTICS_AVAILABLE = False


class UltralyticsYOLOWorldITM:
    """Ultralytics YOLO-World Image-Text Matching - uses detection confidence as similarity score."""

    def __init__(
        self, model_name: str = "yolov8x-worldv2.pt", device: str = "cuda", confidence_threshold: float = 0.0001
    ):
        """
        初始化Ultralytics YOLO-World用于图像-文本匹配(ITM)

        Args:
            model_name: YOLO-World模型名称 (例如 yolov8n-world.pt, yolov8s-world.pt)
            device: 运行设备 ("cpu" 或 "cuda")
            confidence_threshold: 检测的最小置信度阈值
        """
        # 如果ultralytics不可用，则输出警告并返回
        if not ULTRALYTICS_AVAILABLE:
            print("Warning: Ultralytics not available, similarity will return 0.0")
            self.model = None
            return

        try:
            # 加载YOLO-World模型
            print(f"Loading Ultralytics YOLO-World: {model_name} on {device}")
            self.model = YOLO(model_name)
            self.device = device
            self.confidence_threshold = confidence_threshold

            # 添加类别缓存机制
            self._current_classes = None
            self._classes_cache = {}

            # 强制确保模型完全在指定设备上
            if device == "cuda":
                self.model.to("cuda")
                # 强制同步所有模型组件到CUDA
                if hasattr(self.model.model, "model"):
                    self.model.model.model.to("cuda")
            else:
                # CPU模式 - 强制将所有组件移动到CPU
                self.model.to("cpu")
                if hasattr(self.model.model, "model"):
                    self.model.model.model.to("cpu")
                # 清除任何CUDA缓存
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            print(f"✅ UltralyticsYOLOWorldITM loaded on {device}")

        except Exception as e:
            # 如果加载失败，输出错误信息
            print(f"❌ Failed to load Ultralytics YOLO-World model: {e}")
            self.model = None

    def cosine(self, image: np.ndarray, txt: str) -> float:
        """
        计算图像和文本之间的语义相似度，直接使用YOLO-World检测置信度。

        Args:
            image (numpy.ndarray): 输入图像，numpy数组格式(RGB)。
            txt (str): 需要匹配的文本提示。

        Returns:
            float: 语义相似度分数 (0-1)，基于检测置信度。
        """
        print(f"🔍 UltralyticsYOLOWorldITM.cosine called with txt: '{txt}'")

        if self.model is None:
            # 如果模型未加载成功，返回0
            print("❌ Model is None, returning 0.0")
            return 0.0

        try:
            # 如果需要，将RGB转换为BGR用于OpenCV/YOLO
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image

            # 简单直接的置信度计算
            return self._compute_confidence_similarity(image_bgr, txt)

        except Exception as e:
            # 捕获异常并返回0
            print(f"❌ UltralyticsYOLOWorldITM error: {e}")
            import traceback

            traceback.print_exc()
            return 0.0

    def _compute_confidence_similarity(self, image_bgr: np.ndarray, txt: str) -> float:
        """
        使用YOLO-World检测置信度计算语义相似度。
        优化: 避免频繁调用set_classes()
        """
        # 提取文本中的关键词作为检测类别
        keywords = self._extract_keywords(txt)
        print(f"🔍 Keywords extracted: {keywords}")

        if not keywords:
            return 0.0

        # 优化: 只在类别发生变化时才调用set_classes
        keywords_key = tuple(sorted(keywords))  # 创建可哈希的key
        if self._current_classes != keywords_key:
            print(f"🔄 Setting new classes: {keywords}")
            self.model.set_classes(keywords)
            # 强制确保模型在正确的设备上
            if self.device == "cuda":
                self.model.to("cuda")
            else:
                self.model.to("cpu")
            self._current_classes = keywords_key
        else:
            print(f"♻️  Using cached classes: {keywords}")

        # DEBUG: 检查图像输入
        print(f"🖼️  Image shape: {image_bgr.shape}, dtype: {image_bgr.dtype}")
        print(f"🖼️  Image stats: min={image_bgr.min()}, max={image_bgr.max()}, mean={image_bgr.mean():.1f}")

        # 使用YOLO模型预测图像中的对象
        results = self.model.predict(source=image_bgr, conf=self.confidence_threshold, save=False, verbose=False)

        if not results or len(results) == 0:
            print("🎯 No detections found")
            return 0.0

        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            # 获取检测框的置信度
            confidences = result.boxes.conf.cpu().numpy()
            # 返回最高置信度作为相似度分数
            max_confidence = float(np.max(confidences))
            print(f"🎯 Detection confidence: {max_confidence:.3f}")
            return max_confidence

        print("🎯 No confident detections")
        return 0.0

    def _extract_keywords(self, txt: str) -> List[str]:
        """从文本中提取关键词用于检测。"""
        # 简化关键词提取，直接使用文本
        keywords = [txt.strip().lower()]

        return keywords


class UltralyticsYOLOWorldITMClient:
    """Ultralytics YOLO-World ITM服务器的客户端。"""

    def __init__(self, port: int = 12187):
        """
        初始化Ultralytics YOLO-World ITM服务器客户端

        Args:
            port: 服务器端口 (默认12187以避免冲突)
        """
        self.url = f"http://localhost:{port}/ultralytics_yoloworld_itm"
        print(f"UltralyticsYOLOWorldITMClient initialized on port {port}")

    def cosine(self, image: np.ndarray, txt: str) -> float:
        """客户端图像-文本匹配方法。"""
        print(f"📡 UltralyticsYOLOWorldITMClient.cosine: {image.shape}, '{txt}'")
        print(f"📡 Sending request to: {self.url}")
        try:
            # 向服务器发送请求
            response = send_request(self.url, image=image, txt=txt)
            print(f"📡 Server response: {response}")
            # 解析响应中的相似度值
            similarity = float(response["response"])
            print(f"✅ cosine result: {similarity:.3f}")
            return similarity

        except Exception as e:
            # 如果发生错误，输出错误信息并返回0
            print(f"❌ UltralyticsYOLOWorldITMClient error: {e}")
            import traceback

            traceback.print_exc()
            return 0.0


# 服务器托管函数
if __name__ == "__main__":
    import argparse

    from .server_wrapper import ServerMixin, host_model, str_to_image

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Host Ultralytics YOLO-World ITM Server")
    parser.add_argument("--port", type=int, default=12187)
    parser.add_argument("--model", type=str, default="yolov8n-world.pt", help="YOLO-World model name")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to run on")
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold")
    args = parser.parse_args()

    print(f"Loading Ultralytics YOLO-World ITM model: {args.model}")

    # 创建一个继承ServerMixin和UltralyticsYOLOWorldITM的服务器类
    class UltralyticsYOLOWorldITMServer(ServerMixin, UltralyticsYOLOWorldITM):
        def process_payload(self, payload: dict) -> dict:
            # 从payload中提取图像和文本
            image = str_to_image(payload["image"])
            # 计算相似度
            similarity = self.cosine(image, payload["txt"])
            return {"response": similarity}

    # 创建服务器实例
    server = UltralyticsYOLOWorldITMServer(model_name=args.model, device=args.device, confidence_threshold=args.conf)
    print("Model loaded!")
    print(f"Hosting Ultralytics YOLO-World ITM server on port {args.port}...")
    # 托管模型服务
    host_model(server, name="ultralytics_yoloworld_itm", port=args.port)
