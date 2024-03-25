import argparse
import copy
import os
from pathlib import Path
import logging
from collections import OrderedDict
from pprint import pprint
import random

import numpy as np
from omegaconf import OmegaConf, SCMode
import torch
from einops import rearrange, repeat
import cv2
from PIL import Image
from diffusers.models.autoencoder_kl import AutoencoderKL

from mmcm.utils.load_util import load_pyhon_obj
from mmcm.utils.seed_util import set_all_seed
from mmcm.utils.signature import get_signature_of_string
from mmcm.utils.task_util import fiss_tasks, generate_tasks as generate_tasks_from_table
from mmcm.vision.utils.data_type_util import is_video, is_image, read_image_as_5d
from mmcm.utils.str_util import clean_str_for_save
from mmcm.vision.data.video_dataset import DecordVideoDataset
from musev.auto_prompt.util import generate_prompts

from musev.models.controlnet import PoseGuider
from musev.models.facein_loader import load_facein_extractor_and_proj_by_name
from musev.models.referencenet_loader import load_referencenet_by_name
from musev.models.ip_adapter_loader import (
    load_ip_adapter_vision_clip_encoder_by_name,
    load_vision_clip_encoder_by_name,
    load_ip_adapter_image_proj_by_name,
)
from musev.models.ip_adapter_face_loader import (
    load_ip_adapter_face_extractor_and_proj_by_name,
)
from musev.pipelines.pipeline_controlnet_predictor import (
    DiffusersPipelinePredictor,
)
from musev.models.referencenet import ReferenceNet2D
from musev.models.unet_loader import load_unet_by_name
from musev.utils.util import save_videos_grid_with_opencv
from musev import logger

logger.setLevel("INFO")

file_dir = os.path.dirname(__file__)
PROJECT_DIR = os.path.join(os.path.dirname(__file__), "../..")
DATA_DIR = os.path.join(PROJECT_DIR, "data")


# TODO：use group to group arguments
def parse_args():
    parser = argparse.ArgumentParser(description="musev video to video")
    parser.add_argument(
        "-test_data_path",
        type=str,
        help=(
            "Path to the test data configuration file, now only support yaml ext, "
            "task file simialr to musev/configs/tasks/example.yaml"
        ),
    )
    parser.add_argument(
        "--target_datas",
        type=str,
        default="all",
        help="Names of the test data to run, to select sub tasks, default=`all`",
    )
    parser.add_argument(
        "--sd_model_cfg_path",
        type=str,
        default=os.path.join(PROJECT_DIR, "configs/model/T2I_all_model.py"),
        help="Path to the model configuration file",
    )
    parser.add_argument(
        "--sd_model_name",
        type=str,
        default="all",
        help="Names of the models to run, or path.",
    )
    parser.add_argument(
        "--unet_model_cfg_path",
        type=str,
        default=os.path.join(PROJECT_DIR, "./configs/model/motion_model.py"),
        help="Path to motion_cfg path or motion unet path",
    )
    parser.add_argument(
        "--unet_model_name",
        type=str,
        default="musev_referencenet",
        help=(
            "class Name of the unet model, use load_unet_by_name to init unet,"
            "now only support `musev`, `musev_referencenet`,"
        ),
    )
    parser.add_argument(
        "--lcm_model_cfg_path",
        type=str,
        default=os.path.join(PROJECT_DIR, "./configs/model/lcm_model.py"),
        help="Path to lcm lora path",
    )
    parser.add_argument(
        "--lcm_model_name",
        type=str,
        default=None,
        help="lcm model name, None means do not use lcm_lora default=`None`",
        choices=[
            "lcm",
        ],
    )
    parser.add_argument(
        "--referencenet_model_cfg_path",
        type=str,
        default=os.path.join(PROJECT_DIR, "./configs/model/referencenet.py"),
        help="Path to referencenet model config path",
    )
    parser.add_argument(
        "--referencenet_model_name",
        type=str,
        default=None,
        help="referencenet model name, None means do not use referencenet, default=`None`",
        choices=["musev_referencenet"],
    )
    parser.add_argument(
        "--ip_adapter_model_cfg_path",
        type=str,
        default=os.path.join(PROJECT_DIR, "./configs/model/ip_adapter.py"),
        help="Path to ip_adapter model config path",
    )
    parser.add_argument(
        "--ip_adapter_model_name",
        type=str,
        default=None,
        help="ip_adapter model name, None means do not use ip_adapter, default=`None`",
        choices=["musev_referencenet"],
    )
    parser.add_argument(
        "--vision_clip_model_path",
        type=str,
        default="./checkpoints/ip_adapter/models/image_encoder",
        help="vision_clip_extractor_class_name vision_clip_model_path, default=`./checkpoints/ip_adapter/models/image_encoder`",
    )
    parser.add_argument(
        "--vision_clip_extractor_class_name",
        type=str,
        default=None,
        help="vision_clip_extractor_class_name None means according to ip_adapter_model_name, default=`None`",
        choices=["ImageClipVisionFeatureExtractor"],
    )
    parser.add_argument(
        "--facein_model_cfg_path",
        type=str,
        default=os.path.join(PROJECT_DIR, "./configs/model/facein.py"),
        help="Path to facein model config path",
    )
    parser.add_argument(
        "--facein_model_name",
        type=str,
        default=None,
        help="facein model name, None means do not use facein, default=`None`",
    )
    parser.add_argument(
        "--ip_adapter_face_model_cfg_path",
        type=str,
        default=os.path.join(PROJECT_DIR, "./configs/model/ip_adapter.py"),
        help="Path to facein model config path",
    )
    parser.add_argument(
        "--ip_adapter_face_model_name",
        type=str,
        default=None,
        help="facein model name, None means do not use ip_adapter_face, default=`None`",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(PROJECT_DIR, "results"),
        help="Output directory, default=`musev/results`",
    )
    parser.add_argument(
        "--save_filetype",
        type=str,
        default="mp4",
        help="Type of file to save the video, default=`mp4`",
        choices=["gif", "mp4", "webp", "images"],
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        default=False,
        help="more than video, whether save generated video into images, default=`False`",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing files, default=`False`",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed, default=`None`",
    )
    parser.add_argument(
        "--cross_attention_dim",
        type=int,
        default=768,
        help="Cross attention dimension, default=`768`",
    )
    parser.add_argument(
        "--n_batch",
        type=int,
        default=1,
        help="Maximum number of iterations to run, total_frames=n_batch*time_size, default=`1`",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=4,
        help="Frames per second for save video,default is same to of training, default=`4`",
    )
    parser.add_argument(
        "--use_condition_image",
        action="store_false",
        help=(
            "Whether to use the first frame of the test dataset as the initial image, default=`True`"
            "now only support image"
        ),
    )
    parser.add_argument(
        "--fix_condition_images",
        action="store_true",
        help=("Whether to fix condition_image for every shot, default=`False`"),
    )
    parser.add_argument(
        "--redraw_condition_image",
        action="store_true",
        help="Whether to use the redrawn first frame as the initial image, default=`False`",
    )
    parser.add_argument(
        "--need_img_based_video_noise",
        action="store_false",
        help="Whether to use noise based on the initial frame when adding noise to the video, default=`True`",
    )
    parser.add_argument(
        "--img_weight",
        type=float,
        default=1e-3,
        help="Weight of the vision_condtion frame to video noise, default=`1e-3`",
    )
    parser.add_argument(
        "--write_info",
        action="store_true",
        help="Whether to write frame index, default=`False`",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Height of the generated video, 如果没有设置会默认使用condition_image或视频尺寸, default=`None`",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Width of the generated video, 如果没有设置会默认使用原视频尺寸，default=`None`",
    )
    parser.add_argument(
        "--img_length_ratio",
        type=float,
        default=1.0,
        help="ratio to resize target width, target height of generated video, default=`1.0`",
    )

    parser.add_argument(
        "--n_cols",
        type=int,
        default=3,
        help="Number of columns in the output video grid, unused, now",
    )
    parser.add_argument(
        "--time_size",
        type=int,
        default=12,
        help="Number of frames to generate per iteration, same as of training, default=`12`",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default="video_fusion",
        help="Type of noise to add to the video, default=`video_fusion`",
        choices=["video_fusion", "random"],
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="guidance_scale of first frame, default=`7.5`",
    )
    parser.add_argument(
        "--video_guidance_scale",
        type=float,
        default=3.5,
        help="video_guidance_scale of video, 改大运动变化增强、畸形、变色风险增大， default=`3.5`",
    )
    parser.add_argument(
        "--video_guidance_scale_end",
        type=float,
        default=None,
        help="changed video_guidance_scale_end with timesteps, None means unchanged, default=`None`",
    ),
    parser.add_argument(
        "--video_guidance_scale_method",
        type=str,
        default="linear",
        help="generate  changed video_guidance_scale with timesteps, default=`linear`",
        choices=["linear", "two_stage", "three_stage", "fix_two_stage"],
    ),
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="inference steps of first frame redraw, default=`30",
    )
    parser.add_argument(
        "--video_num_inference_steps",
        type=int,
        default=10,
        help="inference steps of video, default=`10`",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.8,
        help="Strength of the redrawn image, default=`0.8`",
    )
    parser.add_argument(
        "--video_strength",
        type=float,
        default=1.0,
        help="Strength of the redrawn video, , 当video2video时，建议设置大点, default=`1.0`",
    )
    parser.add_argument(
        "--negprompt_cfg_path",
        type=str,
        default=os.path.join(PROJECT_DIR, "configs/model/negative_prompt.py"),
        help="Path to the negtive prompt configuration file",
    )
    parser.add_argument(
        "--video_negative_prompt",
        type=str,
        default="V2",
        help="video negative prompt",
    ),
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="V2",
        help="first frame negative prompt",
    ),
    parser.add_argument(
        "--motion_speed",
        type=float,
        default=8.0,
        help="motion speed, sample rate in training stage, default=`8.0`",
    ),
    parser.add_argument(
        "--need_hist_match",
        default=False,
        action="store_true",
        help="wthether hist match video with vis cond, default=`False`",
    ),
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
    )
    parser.add_argument(
        "--add_static_video_prompt",
        action="store_true",
        default=False,
        help="add static_video_prompt in head of prompt",
    )
    parser.add_argument(
        "--n_vision_condition",
        type=int,
        default=1,
        help="num of vision_condition , default=`1`",
    )
    parser.add_argument(
        "--controlnet_name",
        type=str,
        default=None,
        help="controlnet for video2video, if multicontrolnet, use `,` sep, such as `a,b`, default=`None`",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=1,
        help="get one per sample_rate frames from given video, default=`1`",
    )
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default=1.0,
        help="controlnet 的重绘参数, default=`1.0",
    )
    parser.add_argument(
        "--which2video",
        default="video",
        type=str,
        choices=["video", "video_middle"],
        help="用什么驱动生成视频，默认video2video，video影响噪声,当选择 video_middle 时，表示video只是用于提取middle，不参与噪声, default=`video`",
    ),
    parser.add_argument(
        "--end_to_end",
        default=True,
        action="store_false",
        help="是否首尾相连的方式生成长视频, default=`True`",
    ),
    parser.add_argument(
        "--fixed_refer_image",
        action="store_false",
        default=True,
        help="whether fix referencenet image or not,默认使用第一帧作为参考帧， default=`True`",
    )
    parser.add_argument(
        "--fixed_ip_adapter_image",
        action="store_false",
        default=True,
        help="whether fixed_ip_adapter_image or not ,默认使用第一帧作为参考帧， default=`True`",
    )
    parser.add_argument(
        "--fixed_refer_face_image",
        action="store_false",
        default=True,
        help="whether fix facein image or not,默认使用第一帧作为参考帧， default=`True`",
    )
    parser.add_argument(
        "--redraw_condition_image_with_referencenet",
        action="store_false",
        default=True,
        help="重绘首帧的时候是否使用ip_adapter， default=`True`",
    )
    parser.add_argument(
        "--redraw_condition_image_with_ipdapter",
        action="store_false",
        default=True,
        help="重绘首帧的时候是否使用ip_adapter， default=`True`",
    )
    parser.add_argument(
        "--need_video2video",
        action="store_true",
        default=False,
        help="video2video时是否使用原视频垫图，video2video默认True，videomiddle2video、按需使用， default=`False`",
    )

    parser.add_argument(
        "--redraw_condition_image_with_facein",
        action="store_false",
        default=True,
        help="重绘首帧的时候是否使用 facein, default=`True`",
    )
    parser.add_argument(
        "--w_ind_noise",
        default=0.5,
        type=float,
        help="videofusion_noise 中 独立噪声的比例， default=`0.5`",
    )
    parser.add_argument(
        "--ip_adapter_scale",
        default=1.0,
        type=float,
        help="ipadapter权重， default=`1.0`",
    )
    parser.add_argument(
        "--facein_scale",
        default=1.0,
        type=float,
        help="facein 权重， default=`1.0`",
    )
    parser.add_argument(
        "--face_image_path",
        default=None,
        type=str,
        help="face_image_str, default=`None`",
    )
    parser.add_argument(
        "--ipadapter_image_path",
        default=None,
        type=str,
        help="face_image_str, default=`None`",
    )
    parser.add_argument(
        "--referencenet_image_path",
        default=None,
        type=str,
        help="referencenet_image_path, default=`None`",
    )
    parser.add_argument(
        "--vae_model_path",
        default="./checkpoints/vae/sd-vae-ft-mse",
        type=str,
        help="vae path, default=`./checkpoints/vae/sd-vae-ft-mse`",
    )
    parser.add_argument(
        "--redraw_condition_image_with_ip_adapter_face",
        action="store_false",
        default=True,
        help="重绘首帧的时候是否使用 facein, default=`True`",
    )
    parser.add_argument(
        "--ip_adapter_face_scale",
        default=1.0,
        type=float,
        help="ip_adapter face default=`1.0`",
    )
    parser.add_argument(
        "--prompt_only_use_image_prompt",
        action="store_true",
        default=False,
        help="prompt_only_use_image_prompt, 在没有ip_adapter_cross_attn时会替换掉text_prompt_emb, default=`False`",
    )

    parser.add_argument(
        "--pose_guider_model_path",
        type=str,
        default=None,
        help="moore pose_guider，等价于controlnet的功能, default=`None`",
    )
    parser.add_argument(
        "--enable_zero_snr",
        action="store_true",
        default=False,
        help="是否使用 zero_snr的scheduler的参数，包括v_prediction、trailing等, default=`False`",
    )

    parser.add_argument(
        "--record_mid_video_noises",
        action="store_true",
        default=False,
        help="是否在去噪过程中，使用前一片段的后几帧噪声影响后一片段的前几帧噪声, default=`False`",
    )
    parser.add_argument(
        "--record_mid_video_latents",
        action="store_true",
        default=False,
        help="是否在去噪过程中，使用前一片段的后几帧latents影响后一片段的前几帧latents, default=`False`",
    )
    parser.add_argument(
        "--video_overlap",
        default=1,
        type=int,
        help="串行去噪过程，前一片段和后一片段的重合帧数，default=`1`",
    )
    parser.add_argument(
        "--context_schedule",
        default="uniform_v2",
        type=str,
        help="并行去噪的片段生成方式，default=`uniform_v2`",
        choices=["uniform", "uniform_v2"],
    )
    parser.add_argument(
        "--context_frames",
        default=12,
        type=int,
        help="并行去噪过程中Unet一次运算的窗口大小，default=`12`",
    )
    parser.add_argument(
        "--context_stride",
        default=1,
        type=int,
        help="并行去噪过程中Unet一次运算的窗口步进，貌似是无效参数，待去掉，default=`1`",
    )
    parser.add_argument(
        "--context_overlap",
        default=4,
        type=int,
        help="并行去噪过程中Unet一次运算的窗口重合部分，default=`4`",
    )
    parser.add_argument(
        "--context_batch_size",
        default=1,
        type=int,
        help="并行去噪过程中Unet一次运行的窗口数量，增加batch_size，会显著增加显存，default=`1`",
    )
    parser.add_argument(
        "--interpolation_factor",
        default=1,
        type=int,
        help="对最后生成的latent进行超分辨，default=`1`",
    )
    parser.add_argument(
        "--video_is_middle",
        action="store_true",
        default=False,
        help="输入视频是否是middle可视化视频, default=`False`",
    )
    parser.add_argument(
        "--video_has_condition",
        action="store_false",
        default=True,
        help="当输入的视频是middle condition时，且有首帧时，middle序列是否含有首帧, default=`True`",
    )
    parser.add_argument(
        "--need_return_videos",
        action="store_true",
        default=False,
        help="保存视频时是否保存原视频, default=`False`",
    )
    parser.add_argument(
        "--need_return_condition",
        action="store_true",
        default=False,
        help="保存视频时是否保存controlnet_middle, default=`False`",
    )

    parser.add_argument(
        "--n_repeat",
        default=1,
        type=int,
        help="每个测试数据跑几遍，default=`1`",
    )
    args = parser.parse_args()
    return args


args = parse_args()
print("args")
pprint(args.__dict__)
print("\n")

logger.setLevel(args.log_level)
overwrite = args.overwrite
cross_attention_dim = args.cross_attention_dim
time_size = args.time_size  # 一次视频生成的帧数
n_batch = args.n_batch  # 按照time_size的尺寸 生成n_batch次，总帧数 = time_size * n_batch
fps = args.fps
fix_condition_images = args.fix_condition_images
use_condition_image = args.use_condition_image  # 当 test_data 中有图像时，作为初始图像
redraw_condition_image = args.redraw_condition_image  # 用于视频生成的首帧是否使用重绘后的
need_img_based_video_noise = (
    args.need_img_based_video_noise
)  # 视频加噪过程中是否使用首帧 condition_images
img_weight = args.img_weight
height = args.height  # 如果测试数据中没有单独指定宽高，则默认这里
width = args.width  # 如果测试数据中没有单独指定宽高，则默认这里
img_length_ratio = args.img_length_ratio  # 如果测试数据中没有单独指定图像宽高比resize比例，则默认这里
n_cols = args.n_cols
noise_type = args.noise_type
strength = args.strength  # 首帧重绘程度参数
video_guidance_scale = args.video_guidance_scale  # 视频 condition与 uncond的权重参数
guidance_scale = args.guidance_scale  # 时序条件帧 condition与uncond的权重参数
video_num_inference_steps = args.video_num_inference_steps  # 视频迭代次数
num_inference_steps = args.num_inference_steps  # 时序条件帧 重绘参数
seed = args.seed
save_filetype = args.save_filetype
save_images = args.save_images
sd_model_cfg_path = args.sd_model_cfg_path
sd_model_name = (
    args.sd_model_name if args.sd_model_name == "all" else args.sd_model_name.split(",")
)
unet_model_cfg_path = args.unet_model_cfg_path
unet_model_name = args.unet_model_name
test_data_path = args.test_data_path
target_datas = (
    args.target_datas if args.target_datas == "all" else args.target_datas.split(",")
)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16
controlnet_name = args.controlnet_name
controlnet_name_str = controlnet_name
if controlnet_name is not None:
    controlnet_name = controlnet_name.split(",")
    if len(controlnet_name) == 1:
        controlnet_name = controlnet_name[0]

video_strength = args.video_strength  # 视频重绘程度参数
sample_rate = args.sample_rate
controlnet_conditioning_scale = args.controlnet_conditioning_scale

end_to_end = args.end_to_end  # 是否首尾相连生成长视频
control_guidance_start = 0.0
control_guidance_end = 0.5
control_guidance_end = 1.0
negprompt_cfg_path = args.negprompt_cfg_path
video_negative_prompt = args.video_negative_prompt
negative_prompt = args.negative_prompt
motion_speed = args.motion_speed
need_hist_match = args.need_hist_match
video_guidance_scale_end = args.video_guidance_scale_end
video_guidance_scale_method = args.video_guidance_scale_method
add_static_video_prompt = args.add_static_video_prompt
n_vision_condition = args.n_vision_condition
lcm_model_cfg_path = args.lcm_model_cfg_path
lcm_model_name = args.lcm_model_name
referencenet_model_cfg_path = args.referencenet_model_cfg_path
referencenet_model_name = args.referencenet_model_name
ip_adapter_model_cfg_path = args.ip_adapter_model_cfg_path
ip_adapter_model_name = args.ip_adapter_model_name
vision_clip_model_path = args.vision_clip_model_path
vision_clip_extractor_class_name = args.vision_clip_extractor_class_name
facein_model_cfg_path = args.facein_model_cfg_path
facein_model_name = args.facein_model_name
ip_adapter_face_model_cfg_path = args.ip_adapter_face_model_cfg_path
ip_adapter_face_model_name = args.ip_adapter_face_model_name

fixed_refer_image = args.fixed_refer_image
fixed_ip_adapter_image = args.fixed_ip_adapter_image
fixed_refer_face_image = args.fixed_refer_face_image
redraw_condition_image_with_referencenet = args.redraw_condition_image_with_referencenet
redraw_condition_image_with_ipdapter = args.redraw_condition_image_with_ipdapter
redraw_condition_image_with_facein = args.redraw_condition_image_with_facein
redraw_condition_image_with_ip_adapter_face = (
    args.redraw_condition_image_with_ip_adapter_face
)
w_ind_noise = args.w_ind_noise
ip_adapter_scale = args.ip_adapter_scale
facein_scale = args.facein_scale
ip_adapter_face_scale = args.ip_adapter_face_scale
face_image_path = args.face_image_path
ipadapter_image_path = args.ipadapter_image_path
referencenet_image_path = args.referencenet_image_path
vae_model_path = args.vae_model_path
prompt_only_use_image_prompt = args.prompt_only_use_image_prompt
pose_guider_model_path = args.pose_guider_model_path
need_video2video = args.need_video2video
# serial_denoise parameter start
record_mid_video_noises = args.record_mid_video_noises
record_mid_video_latents = args.record_mid_video_latents
video_overlap = args.video_overlap
# serial_denoise parameter end
# parallel_denoise parameter start
context_schedule = args.context_schedule
context_frames = args.context_frames
context_stride = args.context_stride
context_overlap = args.context_overlap
context_batch_size = args.context_batch_size
interpolation_factor = args.interpolation_factor
n_repeat = args.n_repeat

video_is_middle = args.video_is_middle
video_has_condition = args.video_has_condition
need_return_videos = args.need_return_videos
need_return_condition = args.need_return_condition
# parallel_denoise parameter end
need_controlnet = controlnet_name is not None

which2video = args.which2video
if which2video == "video":
    which2video_name = "v2v"
elif which2video == "video_middle":
    which2video_name = "vm2v"
else:
    raise ValueError(
        "which2video only support video, video_middle, but given {which2video}"
    )
b = 1
negative_embedding = [
    ["./checkpoints/embedding/badhandv4.pt", "badhandv4"],
    [
        "./checkpoints/embedding/ng_deepnegative_v1_75t.pt",
        "ng_deepnegative_v1_75t",
    ],
    [
        "./checkpoints/embedding/EasyNegativeV2.safetensors",
        "EasyNegativeV2",
    ],
    [
        "./checkpoints/embedding/bad_prompt_version2-neg.pt",
        "bad_prompt_version2-neg",
    ],
]
prefix_prompt = ""
suffix_prompt = ", beautiful, masterpiece, best quality"
suffix_prompt = ""

if sd_model_name != "None":
    # 使用 cfg_path 里的sd_model_path
    sd_model_params_dict_src = load_pyhon_obj(sd_model_cfg_path, "MODEL_CFG")
    sd_model_params_dict = {
        k: v
        for k, v in sd_model_params_dict_src.items()
        if sd_model_name == "all" or k in sd_model_name
    }
else:
    # 使用命令行给的sd_model_path, 需要单独设置 sd_model_name 为None，
    sd_model_name = os.path.basename(sd_model_cfg_path).split(".")[0]
    sd_model_params_dict = {sd_model_name: {"sd": sd_model_cfg_path}}
    sd_model_params_dict_src = sd_model_params_dict
if len(sd_model_params_dict) == 0:
    raise ValueError(
        "has not target model, please set one of {}".format(
            " ".join(list(sd_model_params_dict_src.keys()))
        )
    )
print("running model, T2I SD")
pprint(sd_model_params_dict)

# lcm
if lcm_model_name is not None:
    lcm_model_params_dict_src = load_pyhon_obj(lcm_model_cfg_path, "MODEL_CFG")
    print("lcm_model_params_dict_src")
    lcm_lora_dct = lcm_model_params_dict_src[lcm_model_name]
else:
    lcm_lora_dct = None
print("lcm: ", lcm_model_name, lcm_lora_dct)


# motion net parameters
if os.path.isdir(unet_model_cfg_path):
    unet_model_path = unet_model_cfg_path
elif os.path.isfile(unet_model_cfg_path):
    unet_model_params_dict_src = load_pyhon_obj(unet_model_cfg_path, "MODEL_CFG")
    print("unet_model_params_dict_src", unet_model_params_dict_src.keys())
    unet_model_path = unet_model_params_dict_src[unet_model_name]["unet"]
else:
    raise ValueError(f"expect dir or file, but given {unet_model_cfg_path}")
print("unet: ", unet_model_name, unet_model_path)


# referencenet
if referencenet_model_name is not None:
    if os.path.isdir(referencenet_model_cfg_path):
        referencenet_model_path = referencenet_model_cfg_path
    elif os.path.isfile(referencenet_model_cfg_path):
        referencenet_model_params_dict_src = load_pyhon_obj(
            referencenet_model_cfg_path, "MODEL_CFG"
        )
        print(
            "referencenet_model_params_dict_src",
            referencenet_model_params_dict_src.keys(),
        )
        referencenet_model_path = referencenet_model_params_dict_src[
            referencenet_model_name
        ]["net"]
    else:
        raise ValueError(f"expect dir or file, but given {referencenet_model_cfg_path}")
else:
    referencenet_model_path = None
print("referencenet: ", referencenet_model_name, referencenet_model_path)


# ip_adapter
if ip_adapter_model_name is not None:
    ip_adapter_model_params_dict_src = load_pyhon_obj(
        ip_adapter_model_cfg_path, "MODEL_CFG"
    )
    print("ip_adapter_model_params_dict_src", ip_adapter_model_params_dict_src.keys())
    ip_adapter_model_params_dict = ip_adapter_model_params_dict_src[
        ip_adapter_model_name
    ]
else:
    ip_adapter_model_params_dict = None
print("ip_adapter: ", ip_adapter_model_name, ip_adapter_model_params_dict)


# facein
if facein_model_name is not None:
    facein_model_params_dict_src = load_pyhon_obj(facein_model_cfg_path, "MODEL_CFG")
    print("facein_model_params_dict_src", facein_model_params_dict_src.keys())
    facein_model_params_dict = facein_model_params_dict_src[facein_model_name]
else:
    facein_model_params_dict = None
print("facein: ", facein_model_name, facein_model_params_dict)

# ip_adapter_face
if ip_adapter_face_model_name is not None:
    ip_adapter_face_model_params_dict_src = load_pyhon_obj(
        ip_adapter_face_model_cfg_path, "MODEL_CFG"
    )
    print(
        "ip_adapter_face_model_params_dict_src",
        ip_adapter_face_model_params_dict_src.keys(),
    )
    ip_adapter_face_model_params_dict = ip_adapter_face_model_params_dict_src[
        ip_adapter_face_model_name
    ]
else:
    ip_adapter_face_model_params_dict = None
print(
    "ip_adapter_face: ", ip_adapter_face_model_name, ip_adapter_face_model_params_dict
)


# negative_prompt
def get_negative_prompt(negative_prompt, cfg_path=None, n: int = 10):
    name = negative_prompt[:n]
    if cfg_path is not None and cfg_path not in ["None", "none"]:
        dct = load_pyhon_obj(cfg_path, "Negative_Prompt_CFG")
        negative_prompt = dct[negative_prompt]["prompt"]

    return name, negative_prompt


negtive_prompt_length = 10
video_negative_prompt_name, video_negative_prompt = get_negative_prompt(
    video_negative_prompt,
    cfg_path=negprompt_cfg_path,
    n=negtive_prompt_length,
)
negative_prompt_name, negative_prompt = get_negative_prompt(
    negative_prompt,
    cfg_path=negprompt_cfg_path,
    n=negtive_prompt_length,
)

print("video_negprompt", video_negative_prompt_name, video_negative_prompt)
print("negprompt", negative_prompt_name, negative_prompt)

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)


# test_data_parameters
def load_yaml(path):
    tasks = OmegaConf.to_container(
        OmegaConf.load(path), structured_config_mode=SCMode.INSTANTIATE, resolve=True
    )
    return tasks


if test_data_path.endswith(".yaml"):
    test_datas_src = load_yaml(test_data_path)
elif test_data_path.endswith(".csv"):
    test_datas_src = generate_tasks_from_table(test_data_path)
else:
    raise ValueError("expect yaml or csv, but given {}".format(test_data_path))

test_datas = [
    test_data
    for test_data in test_datas_src
    if target_datas == "all" or test_data.get("name", None) in target_datas
]

test_datas = fiss_tasks(test_datas)
test_datas = generate_prompts(test_datas)

n_test_datas = len(test_datas)
if n_test_datas == 0:
    raise ValueError(
        "n_test_datas == 0, set target_datas=None or set atleast one of {}".format(
            " ".join(list(d.get("name", "None") for d in test_datas_src))
        )
    )
print("n_test_datas", n_test_datas)
# pprint(test_datas)


def read_image(path):
    name = os.path.basename(path).split(".")[0]
    image = read_image_as_5d(path)
    return image, name


def read_image_lst(path):
    images_names = [read_image(x) for x in path]
    images, names = zip(*images_names)
    images = np.concatenate(images, axis=2)
    name = "_".join(names)
    return images, name


def read_image_and_name(path):
    if isinstance(path, str):
        path = [path]
    images, name = read_image_lst(path)
    return images, name


if referencenet_model_name is not None:
    referencenet = load_referencenet_by_name(
        model_name=referencenet_model_name,
        # sd_model=sd_model_path,
        # sd_model="./checkpoints/Moore-AnimateAnyone/AnimateAnyone/reference_unet.pth",
        sd_referencenet_model=referencenet_model_path,
        cross_attention_dim=cross_attention_dim,
    )
else:
    referencenet = None
    referencenet_model_name = "no"

if vision_clip_extractor_class_name is not None:
    vision_clip_extractor = load_vision_clip_encoder_by_name(
        ip_image_encoder=vision_clip_model_path,
        vision_clip_extractor_class_name=vision_clip_extractor_class_name,
    )
    logger.info(
        f"vision_clip_extractor, name={vision_clip_extractor_class_name}, path={vision_clip_model_path}"
    )
else:
    vision_clip_extractor = None
    logger.info(f"vision_clip_extractor, None")

if ip_adapter_model_name is not None:
    ip_adapter_image_proj = load_ip_adapter_image_proj_by_name(
        model_name=ip_adapter_model_name,
        ip_image_encoder=ip_adapter_model_params_dict.get(
            "ip_image_encoder", vision_clip_model_path
        ),
        ip_ckpt=ip_adapter_model_params_dict["ip_ckpt"],
        cross_attention_dim=cross_attention_dim,
        clip_embeddings_dim=ip_adapter_model_params_dict["clip_embeddings_dim"],
        clip_extra_context_tokens=ip_adapter_model_params_dict[
            "clip_extra_context_tokens"
        ],
        ip_scale=ip_adapter_model_params_dict["ip_scale"],
        device=device,
    )
else:
    ip_adapter_image_proj = None
    ip_adapter_model_name = "no"

if pose_guider_model_path is not None:
    logger.info(f"PoseGuider ={pose_guider_model_path}")
    pose_guider = PoseGuider.from_pretrained(
        pose_guider_model_path,
        conditioning_embedding_channels=320,
        block_out_channels=(16, 32, 96, 256),
    )
else:
    pose_guider = None

for model_name, sd_model_params in sd_model_params_dict.items():
    lora_dict = sd_model_params.get("lora", None)
    model_sex = sd_model_params.get("sex", None)
    model_style = sd_model_params.get("style", None)
    sd_model_path = sd_model_params["sd"]
    test_model_vae_model_path = sd_model_params.get("vae", vae_model_path)

    unet = load_unet_by_name(
        model_name=unet_model_name,
        sd_unet_model=unet_model_path,
        sd_model=sd_model_path,
        # sd_model="./checkpoints/Moore-AnimateAnyone/AnimateAnyone/denoising_unet.pth",
        cross_attention_dim=cross_attention_dim,
        need_t2i_facein=facein_model_name is not None,
        # facein 目前没参与训练，但在unet中定义了，载入相关参数会报错，所以用strict控制
        strict=not (facein_model_name is not None),
        need_t2i_ip_adapter_face=ip_adapter_face_model_name is not None,
    )

    if facein_model_name is not None:
        (
            face_emb_extractor,
            facein_image_proj,
        ) = load_facein_extractor_and_proj_by_name(
            model_name=facein_model_name,
            ip_image_encoder=facein_model_params_dict["ip_image_encoder"],
            ip_ckpt=facein_model_params_dict["ip_ckpt"],
            cross_attention_dim=cross_attention_dim,
            clip_embeddings_dim=facein_model_params_dict["clip_embeddings_dim"],
            clip_extra_context_tokens=facein_model_params_dict[
                "clip_extra_context_tokens"
            ],
            ip_scale=facein_model_params_dict["ip_scale"],
            device=device,
            # facein目前没有参与unet中的训练，需要单独载入参数
            unet=unet,
        )
    else:
        face_emb_extractor = None
        facein_image_proj = None

    if ip_adapter_face_model_name is not None:
        (
            ip_adapter_face_emb_extractor,
            ip_adapter_face_image_proj,
        ) = load_ip_adapter_face_extractor_and_proj_by_name(
            model_name=ip_adapter_face_model_name,
            ip_image_encoder=ip_adapter_face_model_params_dict["ip_image_encoder"],
            ip_ckpt=ip_adapter_face_model_params_dict["ip_ckpt"],
            cross_attention_dim=cross_attention_dim,
            clip_embeddings_dim=ip_adapter_face_model_params_dict[
                "clip_embeddings_dim"
            ],
            clip_extra_context_tokens=ip_adapter_face_model_params_dict[
                "clip_extra_context_tokens"
            ],
            ip_scale=ip_adapter_face_model_params_dict["ip_scale"],
            device=device,
            unet=unet,  # ip_adapter_face 目前没有参与unet中的训练，需要单独载入参数
        )
    else:
        ip_adapter_face_emb_extractor = None
        ip_adapter_face_image_proj = None

    print("test_model_vae_model_path", test_model_vae_model_path)

    sd_predictor = DiffusersPipelinePredictor(
        sd_model_path=sd_model_path,
        unet=unet,
        lora_dict=lora_dict,
        lcm_lora_dct=lcm_lora_dct,
        device=device,
        dtype=torch_dtype,
        negative_embedding=negative_embedding,
        referencenet=referencenet,
        ip_adapter_image_proj=ip_adapter_image_proj,
        vision_clip_extractor=vision_clip_extractor,
        facein_image_proj=facein_image_proj,
        face_emb_extractor=face_emb_extractor,
        vae_model=test_model_vae_model_path,
        ip_adapter_face_emb_extractor=ip_adapter_face_emb_extractor,
        ip_adapter_face_image_proj=ip_adapter_face_image_proj,
        pose_guider=pose_guider,
        controlnet_name=controlnet_name,
        # TODO: 一些过期参数，待去掉
        include_body=True,
        include_face=False,
        include_hand=True,
        enable_zero_snr=args.enable_zero_snr,
    )
    logger.debug(f"load referencenet"),

    for i_test_data, test_data in enumerate(test_datas):
        batch = []
        texts = []
        video_path = test_data.get("video_path")
        video_reader = DecordVideoDataset(
            video_path,
            time_size=time_size,
            step=time_size,
            sample_rate=sample_rate,
            device="cpu",
            data_type="rgb",
            channels_order="c t h w",
            drop_last=True,
        )
        video_height = video_reader.height
        video_width = video_reader.width

        print("\n i_test_data", i_test_data, model_name)
        test_data_name = test_data.get("name", i_test_data)
        prompt = test_data["prompt"]
        prompt = prefix_prompt + prompt + suffix_prompt
        prompt_hash = get_signature_of_string(prompt, length=5)
        test_data["prompt_hash"] = prompt_hash
        test_data_height = test_data.get("height", height)
        test_data_width = test_data.get("width", width)
        test_data_condition_images_path = test_data.get("condition_images", None)
        test_data_condition_images_index = test_data.get("condition_images_index", None)
        test_data_redraw_condition_image = test_data.get(
            "redraw_condition_image", redraw_condition_image
        )
        # read condition_image
        if (
            test_data_condition_images_path is not None
            and use_condition_image
            and (
                isinstance(test_data_condition_images_path, list)
                or (
                    isinstance(test_data_condition_images_path, str)
                    and is_image(test_data_condition_images_path)
                )
            )
        ):
            (
                test_data_condition_images,
                test_data_condition_images_name,
            ) = read_image_and_name(test_data_condition_images_path)
            condition_image_height = test_data_condition_images.shape[3]
            condition_image_width = test_data_condition_images.shape[4]
            logger.debug(
                f"test_data_condition_images use {test_data_condition_images_path}"
            )
        else:
            test_data_condition_images = None
            test_data_condition_images_name = "no"
            condition_image_height = None
            condition_image_width = None
            logger.debug(f"test_data_condition_images is None")

        # 当没有指定生成视频的宽高时，使用输入条件的宽高，优先使用 condition_image，低优使用 video
        if test_data_height is None:
            test_data_height = (
                condition_image_height
                if condition_image_height is not None
                else video_height
            )

        if test_data_width is None:
            test_data_width = (
                condition_image_width
                if condition_image_width is not None
                else video_width
            )

        test_data_img_length_ratio = float(
            test_data.get("img_length_ratio", img_length_ratio)
        )
        test_data_height = int(test_data_height * test_data_img_length_ratio // 64 * 64)
        test_data_width = int(test_data_width * test_data_img_length_ratio // 64 * 64)
        pprint(test_data)
        print(f"test_data_height={test_data_height}")
        print(f"test_data_width={test_data_width}")
        # continue
        test_data_style = test_data.get("style", None)
        test_data_sex = test_data.get("sex", None)
        # 如果使用|进行多参数任务设置时对应的字段是字符串类型，需要显式转换浮点数。
        test_data_motion_speed = float(test_data.get("motion_speed", motion_speed))
        test_data_w_ind_noise = float(test_data.get("w_ind_noise", w_ind_noise))
        test_data_img_weight = float(test_data.get("img_weight", img_weight))
        logger.debug(
            f"test_data_condition_images_path {test_data_condition_images_path}"
        )
        logger.debug(
            f"test_data_condition_images_index {test_data_condition_images_index}"
        )
        test_data_refer_image_path = test_data.get(
            "refer_image", referencenet_image_path
        )
        test_data_ipadapter_image_path = test_data.get(
            "ipadapter_image", ipadapter_image_path
        )
        test_data_refer_face_image_path = test_data.get("face_image", face_image_path)
        test_data_video_is_middle = test_data.get("video_is_middle", video_is_middle)
        test_data_video_has_condition = test_data.get(
            "video_has_condition", video_has_condition
        )

        controlnet_processor_params = {
            "detect_resolution": min(test_data_height, test_data_width),
            "image_resolution": min(test_data_height, test_data_width),
        }
        if negprompt_cfg_path is not None:
            if "video_negative_prompt" in test_data:
                (
                    test_data_video_negative_prompt_name,
                    test_data_video_negative_prompt,
                ) = get_negative_prompt(
                    test_data.get(
                        "video_negative_prompt",
                    ),
                    cfg_path=negprompt_cfg_path,
                    n=negtive_prompt_length,
                )
            else:
                test_data_video_negative_prompt_name = video_negative_prompt_name
                test_data_video_negative_prompt = video_negative_prompt
            if "negative_prompt" in test_data:
                (
                    test_data_negative_prompt_name,
                    test_data_negative_prompt,
                ) = get_negative_prompt(
                    test_data.get(
                        "negative_prompt",
                    ),
                    cfg_path=negprompt_cfg_path,
                    n=negtive_prompt_length,
                )
            else:
                test_data_negative_prompt_name = negative_prompt_name
                test_data_negative_prompt = negative_prompt
        else:
            test_data_video_negative_prompt = test_data.get(
                "video_negative_prompt", video_negative_prompt
            )
            test_data_video_negative_prompt_name = test_data_video_negative_prompt[
                :negtive_prompt_length
            ]
            test_data_negative_prompt = test_data.get(
                "negative_prompt", negative_prompt
            )
            test_data_negative_prompt_name = test_data_negative_prompt[
                :negtive_prompt_length
            ]

        # 准备 test_data_refer_image
        if referencenet is not None:
            if test_data_refer_image_path is None:
                test_data_refer_image = test_data_condition_images
                test_data_refer_image_name = test_data_condition_images_name
                logger.debug(f"test_data_refer_image use test_data_condition_images")
            else:
                test_data_refer_image, test_data_refer_image_name = read_image_and_name(
                    test_data_refer_image_path
                )
                logger.debug(f"test_data_refer_image use {test_data_refer_image_path}")
        else:
            test_data_refer_image = None
            test_data_refer_image_name = "no"
            logger.debug(f"test_data_refer_image is None")

        # 准备 test_data_ipadapter_image
        if vision_clip_extractor is not None:
            if test_data_ipadapter_image_path is None:
                test_data_ipadapter_image = test_data_condition_images
                test_data_ipadapter_image_name = test_data_condition_images_name

                logger.debug(
                    f"test_data_ipadapter_image use test_data_condition_images"
                )
            else:
                (
                    test_data_ipadapter_image,
                    test_data_ipadapter_image_name,
                ) = read_image_and_name(test_data_ipadapter_image_path)
                logger.debug(
                    f"test_data_ipadapter_image use f{test_data_ipadapter_image_path}"
                )
        else:
            test_data_ipadapter_image = None
            test_data_ipadapter_image_name = "no"
            logger.debug(f"test_data_ipadapter_image is None")

        # 准备 test_data_refer_face_image
        if facein_image_proj is not None or ip_adapter_face_image_proj is not None:
            if test_data_refer_face_image_path is None:
                test_data_refer_face_image = test_data_condition_images
                test_data_refer_face_image_name = test_data_condition_images_name

                logger.debug(
                    f"test_data_refer_face_image use test_data_condition_images"
                )
            else:
                (
                    test_data_refer_face_image,
                    test_data_refer_face_image_name,
                ) = read_image_and_name(test_data_refer_face_image_path)
                logger.debug(
                    f"test_data_refer_face_image use f{test_data_refer_face_image_path}"
                )
        else:
            test_data_refer_face_image = None
            test_data_refer_face_image_name = "no"
            logger.debug(f"test_data_refer_face_image is None")

        # 当模型的sex、style与test_data同时存在且不相等时，就跳过这个测试用例
        if (
            model_sex is not None
            and test_data_sex is not None
            and model_sex != test_data_sex
        ) or (
            model_style is not None
            and test_data_style is not None
            and model_style != test_data_style
        ):
            print("model doesnt match test_data")
            print("model name: ", model_name)
            print("test_data: ", test_data)
            continue
        # video
        filename = os.path.basename(video_path).split(".")[0]
        for i_num in range(n_repeat):
            test_data_seed = random.randint(0, 1e8) if seed is None else seed
            cpu_generator, gpu_generator = set_all_seed(test_data_seed)

            save_file_name = (
                f"{which2video_name}_m={model_name}_rm={referencenet_model_name}_c={test_data_name}"
                f"_w={test_data_width}_h={test_data_height}_t={time_size}_n={n_batch}"
                f"_vn={video_num_inference_steps}"
                f"_w={test_data_img_weight}_w={test_data_w_ind_noise}"
                f"_s={test_data_seed}_n={controlnet_name_str}"
                f"_s={strength}_g={guidance_scale}_vs={video_strength}_vg={video_guidance_scale}"
                f"_p={prompt_hash}_{test_data_video_negative_prompt_name[:10]}"
                f"_r={test_data_refer_image_name[:3]}_ip={test_data_refer_image_name[:3]}_f={test_data_refer_face_image_name[:3]}"
            )
            save_file_name = clean_str_for_save(save_file_name)
            output_path = os.path.join(
                output_dir,
                f"{save_file_name}.{save_filetype}",
            )
            if os.path.exists(output_path) and not overwrite:
                print("existed", output_path)
                continue

            if which2video in ["video", "video_middle"]:
                if which2video == "video":
                    need_video2video = True
                (
                    out_videos,
                    out_condition,
                    videos,
                ) = sd_predictor.run_pipe_video2video(
                    video=video_path,
                    time_size=time_size,
                    step=time_size,
                    sample_rate=sample_rate,
                    need_return_videos=need_return_videos,
                    need_return_condition=need_return_condition,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    control_guidance_start=control_guidance_start,
                    control_guidance_end=control_guidance_end,
                    end_to_end=end_to_end,
                    need_video2video=need_video2video,
                    video_strength=video_strength,
                    prompt=prompt,
                    width=test_data_width,
                    height=test_data_height,
                    generator=gpu_generator,
                    noise_type=noise_type,
                    negative_prompt=test_data_negative_prompt,
                    video_negative_prompt=test_data_video_negative_prompt,
                    max_batch_num=n_batch,
                    strength=strength,
                    need_img_based_video_noise=need_img_based_video_noise,
                    video_num_inference_steps=video_num_inference_steps,
                    condition_images=test_data_condition_images,
                    fix_condition_images=fix_condition_images,
                    video_guidance_scale=video_guidance_scale,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    redraw_condition_image=test_data_redraw_condition_image,
                    img_weight=test_data_img_weight,
                    w_ind_noise=test_data_w_ind_noise,
                    n_vision_condition=n_vision_condition,
                    motion_speed=test_data_motion_speed,
                    need_hist_match=need_hist_match,
                    video_guidance_scale_end=video_guidance_scale_end,
                    video_guidance_scale_method=video_guidance_scale_method,
                    vision_condition_latent_index=test_data_condition_images_index,
                    refer_image=test_data_refer_image,
                    fixed_refer_image=fixed_refer_image,
                    redraw_condition_image_with_referencenet=redraw_condition_image_with_referencenet,
                    ip_adapter_image=test_data_ipadapter_image,
                    refer_face_image=test_data_refer_face_image,
                    fixed_refer_face_image=fixed_refer_face_image,
                    facein_scale=facein_scale,
                    redraw_condition_image_with_facein=redraw_condition_image_with_facein,
                    ip_adapter_face_scale=ip_adapter_face_scale,
                    redraw_condition_image_with_ip_adapter_face=redraw_condition_image_with_ip_adapter_face,
                    fixed_ip_adapter_image=fixed_ip_adapter_image,
                    ip_adapter_scale=ip_adapter_scale,
                    redraw_condition_image_with_ipdapter=redraw_condition_image_with_ipdapter,
                    prompt_only_use_image_prompt=prompt_only_use_image_prompt,
                    controlnet_processor_params=controlnet_processor_params,
                    # serial_denoise parameter start
                    record_mid_video_noises=record_mid_video_noises,
                    record_mid_video_latents=record_mid_video_latents,
                    video_overlap=video_overlap,
                    # serial_denoise parameter end
                    # parallel_denoise parameter start
                    context_schedule=context_schedule,
                    context_frames=context_frames,
                    context_stride=context_stride,
                    context_overlap=context_overlap,
                    context_batch_size=context_batch_size,
                    interpolation_factor=interpolation_factor,
                    # parallel_denoise parameter end
                    video_is_middle=test_data_video_is_middle,
                    video_has_condition=test_data_video_has_condition,
                )
            else:
                raise ValueError(
                    f"only support video, videomiddle2video, but given {which2video_name}"
                )
            print("out_videos.shape", out_videos.shape)
            batch = [out_videos]
            texts = ["out"]
            if videos is not None:
                print("videos.shape", videos.shape)
                batch.insert(0, videos / 255.0)
                texts.insert(0, "videos")
            if need_controlnet and out_condition is not None:
                if not isinstance(out_condition, list):
                    print("out_condition", out_condition.shape)
                    batch.append(out_condition / 255.0)
                    texts.append(controlnet_name)
                else:
                    batch.extend([x / 255.0 for x in out_condition])
                    texts.extend(controlnet_name)
            out = np.concatenate(batch, axis=0)
            save_videos_grid_with_opencv(
                out,
                output_path,
                texts=texts,
                fps=fps,
                tensor_order="b c t h w",
                n_cols=n_cols,
                write_info=args.write_info,
                save_filetype=save_filetype,
                save_images=save_images,
            )
            print("Save to", output_path)
            print("\n" * 2)
