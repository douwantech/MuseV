from cog import BasePredictor, Input, Path
import subprocess
import uuid
import os
import requests
import yaml
from PIL import Image
from io import BytesIO

class Predictor(BasePredictor):
    def setup(self) -> None:
        checkpoints_dir = "./checkpoints"
        
        if os.path.exists(checkpoints_dir):
            print("Checkpoints directory already exists. Skipping clone.")
        else:
            command = ["git", "clone", "https://huggingface.co/TMElyralab/MuseV", checkpoints_dir]
            result = subprocess.run(command, check=False)
            if result.returncode == 0:
                print("Repository cloned successfully.")
            else:
                print("Failed to clone the repository.")

    def predict(
        self,
        image_input: str = Input(description="Image URL")  # 修改为str类型
    ) -> Path:
        """Run a single prediction on the model"""
        # 生成一个唯一的UUID
        unique_id = str(uuid.uuid4())

        # 创建新的目录路径
        results_dir = f"/src/results/{unique_id}"
        os.makedirs(results_dir, exist_ok=True)

        # 下载condition_images并确定文件后缀
        response = requests.get(image_input)
        content_type = response.headers['Content-Type']
        if 'image/jpeg' in content_type:
            suffix = 'jpg'
        elif 'image/png' in content_type:
            suffix = 'png'
        else:
            suffix = 'jpg'  # 默认后缀

        # 保存下载的图片
        image_path = os.path.join(results_dir, f"condition_image.{suffix}")
        with open(image_path, 'wb') as file:
            file.write(response.content)

        # 从下载的图片中获取宽度和高度
        image = Image.open(BytesIO(response.content))
        width, height = image.size

        # 创建新的data.yaml文件内容
        data = [{
            "condition_images": image_path,
            "eye_blinks_factor": 1.8,
            "height": height,
            "img_length_ratio": 0.957,
            "ipadapter_image": image_path,
            "name": unique_id,
            "prompt": "(masterpiece, best quality, highres:1),(1boy, solo:1),(eye blinks:1.8),(head wave:1.3)",
            "refer_image": image_path,
            "video_path": None,
            "width": width
        }]

        # 写入新的data.yaml文件
        data_yaml_path = os.path.join(results_dir, "data.yaml")
        with open(data_yaml_path, "w") as file:
            yaml.dump(data, file)

        os.environ['PYTHONPATH'] = '/src:/src/MMCM:/src/diffusers/src:/src/controlnet_aux/src'

        # 定义命令及其参数
        command = [
            "python", "scripts/inference/text2video.py",
            "--sd_model_name", "majicmixRealv6Fp16",
            "--unet_model_name", "musev_referencenet",
            "--referencenet_model_name", "musev_referencenet",
            "--ip_adapter_model_name", "musev_referencenet",
            "-test_data_path", data_yaml_path,
            "--output_dir", results_dir,
            "--n_batch", "1",
            "--target_datas", unique_id,
            "--vision_clip_extractor_class_name", "ImageClipVisionFeatureExtractor",
            "--vision_clip_model_path", "./checkpoints/IP-Adapter/models/image_encoder",
            "--time_size", "12",
            "--fps", "12"
        ]

        # 执行命令
        result = subprocess.run(command, capture_output=True, text=True)
        
        # 输出结果
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        # 搜索目录中的mp4文件并重命名为result.mp4
        for file_name in os.listdir(results_dir):
            if file_name.endswith('.mp4'):
                mp4_path = os.path.join(results_dir, file_name)
                new_mp4_path = os.path.join(results_dir, "result.mp4")
                os.rename(mp4_path, new_mp4_path)
                return Path(new_mp4_path)

        return Path("/src/output")