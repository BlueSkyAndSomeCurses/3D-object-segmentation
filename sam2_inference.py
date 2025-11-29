from sam2.build_sam import build_sam2_video_predictor
import torch
import numpy as np
import cv2

from pathlib import Path
import json

np.random.seed(42)
torch.manual_seed(42)


def run_sam2_inference(
    images_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    model_cfg_path: str = "configs/sam2.1/sam2.1_hiera_s.yaml",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)

    predictor = build_sam2_video_predictor(
        model_cfg_path, checkpoint_path, device=device
    )

    image_files = sorted([f for f in images_path.iterdir()])
    first_frame = cv2.imread(image_files[0])
    height, width = first_frame.shape[:2]
    roi = cv2.selectROI(
        "Select Object to Track", first_frame, fromCenter=False, showCrosshair=True
    )
    cv2.destroyAllWindows()

    roi_box = np.array([roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3]])

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        inference_state = predictor.init_state(str(images_path))
        _, object_ids, masks = predictor.add_new_points_or_box(
            inference_state, frame_idx=0, obj_id=1, box=roi_box
        )

        for frame_idx, object_ids, masks in predictor.propagate_in_video(
            inference_state
        ):
            image_path = image_files[frame_idx]
            image = cv2.imread(str(image_path))
            height, width = image.shape[:2]

            print(
                f"Processing frame {frame_idx + 1}/{len(image_files)}: {image_path.name}"
            )

            masks_data = []
            for obj_idx, obj_id in enumerate(object_ids):
                mask = masks[obj_idx, 0].cpu().numpy()
                binary_mask = (mask > 0.5).astype(np.uint8)

                y_coords, x_coords = np.where(binary_mask)
                pixel_pairs = [[int(x), int(y)] for x, y in zip(x_coords, y_coords)]

                if len(x_coords) > 0:
                    bbox = [
                        float(x_coords.min()),
                        float(y_coords.min()),
                        float(x_coords.max() - x_coords.min()),
                        float(y_coords.max() - y_coords.min()),
                    ]
                else:
                    bbox = [0.0, 0.0, 0.0, 0.0]

                mask_dict = {
                    "object_id": int(obj_id),
                    "pixels": pixel_pairs,
                    "area": int(binary_mask.sum()),
                    "bbox": bbox,
                }
                masks_data.append(mask_dict)

            output_masks_path = output_path / f"{image_path.stem}_masks.json"
            with open(output_masks_path, "w") as f:
                json.dump(
                    {
                        "image_name": image_path.name,
                        "image_size": {"height": height, "width": width},
                        "num_objects": len(masks_data),
                        "objects": masks_data,
                    },
                    f,
                )

            print(f"  Saved {len(masks_data)} objects to {output_masks_path.name}")
        predictor.reset_state(inference_state)


if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    run_sam2_inference(
        images_path=Path("./data/frames") / "jpg_frames",
        output_path=Path("./data/sam2_masks"),
        checkpoint_path=Path("./external/sam2/checkpoints/sam2.1_hiera_small.pt"),
        device=device,
    )
