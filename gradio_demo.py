import torchvision.transforms as T
from models import build_model
import os
import torch
import misc as utils
import numpy as np
import torch.nn.functional as F
import argparse
import matplotlib.colors
from torchvision.io import read_video
import torchvision.transforms.functional as Func
from ruamel.yaml import YAML
from easydict import EasyDict
from misc import nested_tensor_from_videos_list
from torch.cuda.amp import autocast
from PIL import Image, ImageDraw, ImageFont
from rich.progress import track
import imageio.v3 as iio
import cv2
import warnings
import gradio as gr
import tempfile
import subprocess
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

transform = T.Compose([
    T.Resize(360),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# colormap
color_list = utils.colormap()
color_list = color_list.astype('uint8').tolist()

def convert_to_mp4(input_path):
    tmp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp_file.name
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        tmp_path
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    return tmp_path

def infer(model, video, text, args):
    assert os.path.exists(video)
    video_name = video.split('/')[-1]
    exp = " ".join(text.lower().split())

    save_name = args.save_name if  args.save_name is not None else video_name
    if not save_name.endswith('.mp4'):
        save_name += '.mp4'

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        # clean output dir
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))
    save_video = os.path.join(output_dir, save_name)

    path = video.name if hasattr(video, 'name') else video
    try:
        safe_path = convert_to_mp4(path)  # optional if codec issues
        video_frames, _, info = read_video(safe_path, pts_unit='sec')
    except (OSError, RuntimeError) as e:
        raise RuntimeError(f"Failed to read video. Original error: {e}")
    
    frames = []
    for i in range(0, len(video_frames), args.frame_step):
        source_frame = Func.to_pil_image(video_frames[i].permute(2, 0, 1))
        frames.append(source_frame) #(C H W)

    video_len = len(frames)

    frames_ids = [x for x in range(video_len)]
    imgs = []
    for t in frames_ids:
        img = frames[t]
        origin_w, origin_h = img.size
        imgs.append(transform(img))

    imgs = torch.stack(imgs, dim=0).to(args.device)
    samples = nested_tensor_from_videos_list(imgs[None], size_divisibility=1)
    img_h, img_w = imgs.shape[-2:]
    size = torch.as_tensor([int(img_h), int(img_w)]).to(args.device)
    target = {"size": size}

    print("begin inference")
    with torch.no_grad():
        with autocast(args.enable_amp):
            outputs = model.infer(samples, [exp], [target])

    pred_logits = outputs["pred_logits"][0]  # [t, q, k]
    pred_masks = outputs["pred_masks"][0]  # [t, q, h, w]
    pred_boxes = outputs["pred_boxes"][0]  # [t, q, 4]

    # according to pred_logits, select the query index
    pred_scores = pred_logits.sigmoid()  # [t, q, k]
    pred_scores = pred_scores.mean(0)  # [q, K]
    max_scores, _ = pred_scores.max(-1)  # [q,]
    _, max_ind = max_scores.max(-1)  # [1,]
    max_inds = max_ind.repeat(video_len)
    pred_masks = pred_masks[range(video_len), max_inds, ...]  # [t, h, w]
    pred_masks = pred_masks.unsqueeze(0)
    pred_boxes = pred_boxes[range(video_len), max_inds].cpu().numpy()  # [t, 4]

    # unpad
    pred_masks = pred_masks[:, :, :img_h, :img_w].cpu()
    pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False)
    pred_masks = (pred_masks.sigmoid() > 0.5).squeeze(0).cpu().numpy()  # 0.5

    print("saving")

    color = "#DC143C"
    color = (np.array(matplotlib.colors.hex2color(color)) * 255).astype('uint8')

    save_imgs = []
    for t, img in enumerate(frames):
        # draw mask
        img = vis_add_mask(img, pred_masks[t], color, args.mask_edge_width)

        draw = ImageDraw.Draw(img)
        draw_boxes = pred_boxes[t][None]
        draw_boxes = rescale_bboxes(draw_boxes, (origin_w, origin_h)).tolist()

        # draw box
        if args.show_box:
            xmin, ymin, xmax, ymax = draw_boxes[0]
            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=tuple(color), width=5)

        # save
        save_dir = os.path.join(output_dir, '{:05d}.png'.format(t))
        if args.save_images:
            img.save(save_dir)
        save_imgs.append(np.asarray(img).copy())

    fps = int(info['video_fps'] / args.frame_step)

    import subprocess
    output_dir = 'tmp_frames'
    os.makedirs(output_dir, exist_ok=True)

    # Save individual frames
    for i, frame in enumerate(save_imgs):
        cv2.imwrite(os.path.join(output_dir, f"{i:05d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # Create MP4 using ffmpeg with H.264 codec
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(output_dir, "%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        save_video
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    # remove temporary frames
    for f in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, f))
    os.rmdir(output_dir)
    print(f"Video created using FFmpeg and saved to {save_video}")
    return save_video


# Post-process functions
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    b = np.stack([
        x_c - 0.5 * w,
        y_c - 0.5 * h,
        x_c + 0.5 * w,
        y_c + 0.5 * h
    ], axis=1)
    return b

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    return b

def vis_add_mask(img, mask, color, edge_width=3):
    origin_img = np.asarray(img.convert('RGB')).copy()
    color = np.array(color)

    mask = mask.reshape(mask.shape[0], mask.shape[1]).astype('uint8') # np
    mask = mask > 0.5

    # Increase the edge width using dilation
    kernel = np.ones((edge_width, edge_width), np.uint8)
    mask_dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    edge_mask = mask_dilated & ~mask

    origin_img[mask] = origin_img[mask] * 0.5 + color * 0.5
    origin_img[edge_mask] = color
    origin_img = Image.fromarray(origin_img)
    return origin_img

def load_model(args):
    model, _, _ = build_model(args)
    device = args.device
    model.to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# -------------------------------
# Inference function for Gradio
# -------------------------------
def rvos_infer(video_file, prompt, checkpoint_path, config_path, device="cuda", frame_step=1, show_box=False):
    print(video_file)
    
    with open(config_path) as f:
        yaml = YAML(typ='safe', pure=True)
        config = yaml.load(f)
    config = {k: v['value'] for k, v in config.items()}
    
    args = {**config, 
            "checkpoint_path": checkpoint_path,
            "device": device,
            "frame_step": frame_step,
            "show_box": show_box,
            "output_dir": "output/gradio_demo",
            "mask_edge_width": 6,
            "save_images": False,
            "save_name": None,
            "tracking_alpha": 0.1,
            "enable_amp": True
           }
    args = EasyDict(args)
    args.GroundingDINO.tracking_alpha = args.tracking_alpha
    
    model = load_model(args)
    save_video = infer(model, video_file, prompt, args)
    # clean torch cache
    torch.cuda.empty_cache()
    return save_video
    
# -------------------------------
# Gradio UI
# -------------------------------
demo = gr.Interface(
    fn=rvos_infer,
    inputs=[
        gr.Video(label="Input Video"),
        gr.Textbox(label="Text Prompt"),
        gr.Textbox(label="Checkpoint Path", placeholder="e.g., checkpoints/dino_ckpt.pth"),
        gr.Textbox(label="Config Path", placeholder="e.g., configs/ytvos_swinb.yaml"),
        gr.Dropdown(["cpu", "cuda"], label="Device", value="cuda"),
        gr.Slider(1, 5, value=1, step=1, label="Frame Step"),
        gr.Checkbox(label="Show Boxes", value=False)
    ],
    outputs=gr.Video(label="Output Video"),
    title="RVOS DINO Video Segmentation",
    description="Upload a video and a text prompt, and RVOS DINO will segment the object described."
)

demo.launch(share=True)
