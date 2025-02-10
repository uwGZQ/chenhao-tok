import os

os.environ["TORCH_HOME"] = "/weka/jieyuz/ziqigao/.cache/torch"
os.environ["HF_HOME"] = "/weka/jieyuz/ziqigao/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/weka/jieyuz/ziqigao/.cache/huggingface"

from torchvision import transforms
from torchvision.transforms import InterpolationMode
import numpy as np
import torch
import cv2
import os
import decord
import sys
import argparse
import glob
from tqdm import tqdm
import pickle

sys.path.append("/weka/jieyuz/ziqigao/iccv_chenhao/videotok/generate_graph/segment_anything_2")
sys.path.append("/weka/jieyuz/ziqigao/iccv_chenhao/videotok/entry")
sys.path.append("/weka/jieyuz/ziqigao/iccv_chenhao/videotok")
sys.path.append("/weka/jieyuz/ziqigao/iccv_chenhao")
from videotok.generate_graph.segment_anything_2.pipeline import TrajGenPipeline
from videotok.entry.share_models.traj_transformer import VideoTokenViT
from videotok.entry.share_models.vit3d import ViT3D

import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def init_traj_vit(model_name='vit-base', ):
    """ init our model """
    config = {
        "model_name"     : model_name,
        "out_channel"    : 64,
        "app_perceiver"  : True,
        "pretrained"     : False,
        "pool"           : "cls",
        "hiera_feat_pool": "sum"
    }
    pos_config = {
        "model_type"      : "perceiver",  # mlp, sincos
        "use_bounding_box": True,
    }
    perceiver_config = {
        "num_latent": 1,
        "depth"     : 2,
        "use_rotary": True
    }
    model = VideoTokenViT(config=config, pos_config=pos_config, perceiver_config=perceiver_config, norm_layer=None)
    return model


def create_transform(image_size):
    """ create image transform """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    normalize = transforms.Normalize(mean, std)
    type_transform = transforms.Lambda(lambda x: x.float().div(255.))

    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BICUBIC),
        type_transform,
        normalize,
    ])


def load_pretrained(
        encoder,
        pretrained,
        checkpoint_key='model',
):
    logger.info(f'Loading pretrained model from {pretrained}')
    checkpoint = torch.load(pretrained, map_location='cpu', weights_only=False)

    pretrained_dict = checkpoint[checkpoint_key]

    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}

    m_pretrained_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith('vision_encoder.vision_encoder.'):
            k = k[len('vision_encoder.'):]
        elif k.startswith('text_encoder'):
            continue
        else:
            k = k.replace('vision_encoder.', '')
        m_pretrained_dict[k] = v
    pretrained_dict = m_pretrained_dict

    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v

    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f'loaded pretrained encoder with msg: {msg}')
    logger.info(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]}\n path: {pretrained}')
    del checkpoint
    return encoder


def preprocess_mask_and_graph(masks, graphs, image_size=224, mask_res_down_factor=4):
    """ resize mask before doing inference.  mask: tensor of shape (T,H,W)"""

    def resize_masks(masks, size):
        T = masks.shape[0]
        resized_masks = np.empty((T, size[0], size[1]), dtype=masks.dtype)
        for t in range(T): resized_masks[t] = cv2.resize(masks[t], (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        return resized_masks

    masks = resize_masks(masks, size=(image_size // mask_res_down_factor, image_size // mask_res_down_factor))
    masks = torch.from_numpy(masks)
    graphs = torch.from_numpy(graphs)

    masks[masks > graphs.max()] = 0
    graphs[graphs > masks.max()] = 0  # some seg idx is invalid due to vanished segmentation size
    graphs = graphs[~(graphs == 0).all(dim=1)]  # filter out rows that are all 0
    return masks, graphs


# a = pickle.load(open('/mnt/sda1/video/activitynet_videos_feat_vittok/v_---9CpRcKoU.pkl', 'rb'))
# b = pickle.load(open('/mnt/sdd1/dynabench/video/Video-ChatGPT/data/ActivityNet_Train_Video-ChatGPT_Clip-L14_Features/activity_clip-14L_spatio_temporal_356/v_---9CpRcKoU.pkl', 'rb'))
# a=1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process your video with given hyperparameters."
    )
    parser.add_argument("--video_folder", type=str, default="/weka/jieyuz/ziqigao/iccv_chenhao/activitynet_videos",
                        help="A folder with video files")
    parser.add_argument("--save_traj_folder", type=str, default="/results/activitynet_videos_traj",
                        help="A folder with video files")
    parser.add_argument("--save_feat_folder", type=str, default="/results/activitynet_videos_feat_vittok",
                        help="A folder with video files")

    parser.add_argument("--ckpt_path", type=str, default='/weka/jieyuz/ziqigao/iccv_chenhao/vittok_ckpts/ckpt_24.pth',
                        help="path to model checkpoint")
    parser.add_argument("--model", type=str, choices=["vittok", "vit3d"], default='vittok',
                        help="Select which method to use")

    args = parser.parse_args()

    # hyper-parameters
    num_frames = 16
    traj_gen_resolution = 512
    inference_resolution = 224
    mask_res_down_factor = 4  # set to 2 in test time to increase performance

    model_name = 'vit-large'

    if args.model == 'vittok':
        vittok = init_traj_vit(model_name=model_name)
        model = load_pretrained(vittok, pretrained=args.ckpt_path).eval().cuda()

    else:
        vit3d = ViT3D(img_size=inference_resolution, model_name=model_name, norm_layer=None)
        model = load_pretrained(vit3d, pretrained=args.ckpt_path).eval().cuda()

    ######################## First step: pre-generate all the trajectories and save them to disk #################################
    trajGenModel = TrajGenPipeline(
        sam2_checkpoint="videotok/generate_graph/segment_anything_2/checkpoints/sam2_hiera_small.pt",
        image_size=traj_gen_resolution,
        frame_num=num_frames,
        track_image_size=traj_gen_resolution,
    )

    # init image transformation (Test time)
    transform = create_transform(image_size=inference_resolution)

    raw_video_paths = glob.glob(os.path.join(args.video_folder, f"*"))
    os.makedirs(args.save_traj_folder, exist_ok=True)
    os.makedirs(args.save_feat_folder, exist_ok=True)
    failure = 0
    for video_path in tqdm(raw_video_paths):

        save_video_path = os.path.join(args.save_traj_folder, os.path.basename(video_path))
        extension = os.path.splitext(os.path.basename(save_video_path))[1]

        save_feat_path = os.path.join(args.save_feat_folder, os.path.basename(video_path).replace(extension, '.pkl'))
        if os.path.exists(save_feat_path):
            continue

        if not os.path.exists(save_video_path):
            try:

                sample_video = trajGenModel.sample_and_save_short_video(video_path=video_path, save_video_path=save_video_path)

                trajGenModel.traj_generation(
                    video=sample_video,
                    video_path=save_video_path,
                    use_key_frame_selection=True,
                    visualize=False,
                    save_to_disk=True
                )

                # load inputs
                video_reader = decord.VideoReader(save_video_path, num_threads=1)
                video = video_reader.get_batch(range(len(video_reader))).asnumpy()

                masks = np.load(save_video_path.replace(extension, f'_mask.npz'))['arr_0']
                graphs = np.load(save_video_path.replace(extension, f'_graph.npz'))['tensor']

                # preprocess
                video = torch.from_numpy(video).permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
                input_video = transform(video)
                input_mask, input_graph = preprocess_mask_and_graph(masks, graphs, image_size=inference_resolution, mask_res_down_factor=mask_res_down_factor)

                # make batch dimension & move to gpu
                input_video = input_video.unsqueeze(0).cuda()
                input_mask = input_mask.unsqueeze(0).cuda()
                input_graph = input_graph.unsqueeze(0).cuda()

                # model inference
                with torch.no_grad():
                    if args.model == 'vittok':
                        features = vittok(input_video, segmask=input_mask, video_graph=input_graph)
                    else:
                        features = vit3d(input_video)

                # conver to numpy and save in pickle
                features = features.squeeze().half().cpu().numpy()

                pickle.dump(features, open(save_feat_path, 'wb'))

                print(features.shape)

            except Exception as e:
                print(f"Error: {e}")
                failure += 1
                continue


    print(f"Failure: {failure}")
