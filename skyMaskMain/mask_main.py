import cv2
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
from pathlib import Path

from skyMaskMain.rich_utils import CONSOLE
from skyMaskMain.utils_mask import setup_mask, get_bounding_boxes, get_masks, disp_mask, process_images

class MaskProcessor:
  def __init__(
    self,
    data_dir: Path,
    prompt: str = "sky",
    inspect: bool = False,
  ):
    self.data_dir = Path(data_dir)
    self.prompt = prompt
    self.inspect = inspect

  def mask_loop(self, image_paths, predictor, processor, dino, bt, tt):
    root = self.data_dir.parent
    save_dir = root / "masks"
    save_dir.mkdir(parents=True, exist_ok=True)

    for idx, img_path in tqdm(enumerate(image_paths), total=len(image_paths), desc="Producing Masks", colour='GREEN', disable=False):
        # Load image
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        boxes = get_bounding_boxes(image_pil, self.prompt, processor, dino, bt, tt)

        bool_mask = ~get_masks(image_pil, boxes, predictor).astype(bool)

        h, w = bool_mask.shape[:2]
        if bool_mask.sum() < 0.001 * h * w:
            continue
        
        # Mask the image
        alpha = (bool_mask.astype(np.uint8)) * 255
        rgba = np.dstack([image_rgb, alpha])

        filename = os.path.basename(img_path)
        mask_path = f'{save_dir}/{filename}'

        # Show mask
        #self.inspect = True
        if self.inspect and idx % 10 == 0:
            disp_mask(image_rgb, rgba)

        # transparent pad for top border
        pad = 50
        h, w = image.shape[:2]
        canvas = np.zeros(
            (h + pad, w, 4),  # 4 channels = RGBA
            dtype=np.uint8
        )
        
        canvas[pad:pad+h, :w] = rgba[:, :, [2, 1, 0, 3]]

        # Save mask
        cv2.imwrite(mask_path, canvas) 

    return save_dir

# main/driver function
  def run_mask_processing(self, bt, tt):
    image_paths= process_images(self.data_dir)
    predictor, processor, dino = setup_mask(self.data_dir)
    save_dir = self.mask_loop(image_paths, predictor, processor, dino, bt, tt)
    mask_dir = Path(save_dir).resolve()
    linked_name = f"[link=file://{mask_dir}]{mask_dir}[/link]"
    CONSOLE.log(f"🎉 Finished! 🎉")
    CONSOLE.print(f"Inspect masks:", linked_name)