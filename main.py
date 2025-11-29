import marimo

__generated_with = "0.18.1"
app = marimo.App()


@app.cell
def _():
    import cv2
    import pycolmap
    import numpy as np
    import open3d as o3d

    from pathlib import Path
    import torch
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from typing import Optional
    import matplotlib.pyplot as plt

    import polars as pl
    import plotly.express as px
    from collections import defaultdict

    from tqdm import tqdm
    import json
    return (
        Optional,
        Path,
        cv2,
        defaultdict,
        json,
        np,
        o3d,
        pl,
        plt,
        px,
        pycolmap,
        torch,
        tqdm,
    )


@app.cell
def _(np, torch):
    np.random.seed(42)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return (device,)


@app.cell
def _(device):
    print(device)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # 3D reconstruction
    """)
    return


@app.cell
def _(Path):
    video_path =  Path("./data/IMG_2822.mp4")
    return (video_path,)


@app.cell
def _(Path, cv2, np):
    def extract_frames(video_path: Path, frame_interval: int = 8) -> list[np.ndarray]:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video: {total_frames} frames, {fps:.2f} FPS")

        frame_count = 0
        saved_coun = 0

        frames = []

        while True:
            ret, frame = cap.read()


            if not ret:
                break

            if frame_count % frame_interval == 0:
                rotated_frame = np.rot90(frame, k=3)
                frames.append(rotated_frame)

            frame_count += 1

        cap.release()

        return frames
    return (extract_frames,)


@app.cell
def _(extract_frames, video_path):
    frames = extract_frames(video_path, frame_interval=2)
    return (frames,)


@app.cell
def _(Path, cv2, np):
    def save_frames(frames: list[np.ndarray], output_dir: Path) -> None:
        (output_dir / "png_frames").mkdir(parents=True, exist_ok=True)
        (output_dir / "jpg_frames").mkdir(parents=True, exist_ok=True)

        for idx, frame in enumerate(frames):
            png_frame_path = output_dir / "png_frames" / f"frame_{idx:04d}.png"
            jpg_frame_path = output_dir / "jpg_frames" / f"{idx:04d}.jpg"
            cv2.imwrite(str(png_frame_path), frame)
            cv2.imwrite(str(jpg_frame_path), frame)
    return (save_frames,)


@app.cell
def _(Path, frames, save_frames):
    images_path = Path("./data/frames")
    save_frames(frames, images_path)
    return (images_path,)


@app.cell
def _(Path, images_path, pycolmap):
    database_path = Path("./reconstruction") / "database.db"
    pycolmap.extract_features(database_path = database_path, image_path = images_path / "png_frames")
    return (database_path,)


@app.cell
def _(database_path, pycolmap):
    pycolmap.match_sequential(database_path = database_path)
    return


@app.cell
def _(Path):
    reconstruction_path = Path("./reconstruction")
    reconstruction_path.mkdir(parents = True, exist_ok = True)
    return (reconstruction_path,)


@app.cell
def _(database_path, images_path, pycolmap, reconstruction_path):
    reconstructions = pycolmap.incremental_mapping(database_path = database_path, image_path = images_path / "png_frames", output_path = reconstruction_path)
    return (reconstructions,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Converting to open3d
    """)
    return


@app.cell
def _(reconstruction_path, reconstructions):
    reconstructions[0].export_PLY(reconstruction_path / "model.ply")
    return


@app.cell
def _(Path, o3d):
    asuka_model_path = Path("./reconstruction/model.ply")
    asuka_3d =  o3d.io.read_point_cloud(str(asuka_model_path))
    return (asuka_3d,)


@app.cell
def _(asuka_3d, o3d):
    o3d.visualization.draw_geometries([asuka_3d])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Loading reconstruction
    """)
    return


@app.cell
def _(pycolmap):
    asuka_best_reconstruction = pycolmap.Reconstruction("./reconstruction/0")
    return (asuka_best_reconstruction,)


@app.cell
def _(mo):
    mo.md(r"""
    # SAM2 mask verification
    """)
    return


@app.cell
def _(Path, json):
    def load_sam2_masks(json_path: Path) -> dict:
        with  json_path.open('r') as f:
            return json.load(f)
    return (load_sam2_masks,)


@app.cell
def _(np):
    def pixels_to_mask(pixels: list[list[int]], height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        for x,y in pixels:
            if 0 <= y < height and 0 <= x < width:
                mask[y, x] = 1
        return mask
    return (pixels_to_mask,)


@app.cell
def _(Optional, Path, cv2, load_sam2_masks, np, pixels_to_mask):
    def draw_masks_on_image(
        image: np.ndarray,
        masks_json_path: Path,
        alpha: float = 0.5,
        colors: Optional[list[tuple]] = None,
        draw_contours: bool = True,
        contour_thickness: int = 2
    ) -> np.ndarray:

        masks_data = load_sam2_masks(masks_json_path)
        height = masks_data['image_size']['height']
        width = masks_data['image_size']['width']
        objects = masks_data['objects']
        num_objects = masks_data['num_objects']

        result = image.copy().astype(np.float32)

        if colors is None:
            colors = [
                tuple(np.random.randint(0, 255, 3).tolist())
                for _ in range(num_objects)
            ]

        overlay = image.copy()

        for obj in objects:
            obj_id = obj['object_id']
            pixels = obj['pixels']

            if len(pixels) == 0:
                continue

            color = colors[obj_id % len(colors)]

            mask = pixels_to_mask(pixels, height, width)

            for c in range(3):
                overlay[:, :, c] = np.where(
                    mask == 1,
                    color[c],
                    overlay[:, :, c]
                )

            if draw_contours:
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(result, contours, -1, color, contour_thickness)

        result = cv2.addWeighted(
            overlay.astype(np.float32), alpha,
            result, 1 - alpha,
            0
        ).astype(np.uint8)

        return result
    return (draw_masks_on_image,)


@app.cell
def _(cv2, draw_masks_on_image, plt):
    def visualize_image_with_masks(
        image_path: str,
        masks_json_path: str,
        show: bool = True,
        **kwargs
    ) -> None:

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        result = draw_masks_on_image(image, masks_json_path, **kwargs)


        if show:
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(12, 8))
            plt.imshow(result_rgb)
            plt.title(f"Image with SAM2 Masks")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

    return (visualize_image_with_masks,)


@app.cell
def _(frames, mo):
    frame_num = mo.ui.slider(start = 0, stop = len(frames) - 1, step = 1, label = "Frame Number")
    frame_num
    return (frame_num,)


@app.cell
def _(Path, frame_num, images_path, visualize_image_with_masks):
    image_with_mask = visualize_image_with_masks(
        image_path = str(images_path /"png_frames"/ f"frame_{frame_num.value:04d}.png"),
        masks_json_path = Path(f"./data/sam2_masks/{frame_num.value:04d}_masks.json"),

    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    # 2D -> 3D masks mapping
    """)
    return


@app.cell
def _(mo, np, o3d, pl, px):
    def draw_3d_model(pcd: o3d.geometry.PointCloud) -> None:
        points_np = np.asarray(pcd.points)
        colors_np = np.asarray(pcd.colors)

        df_pcd = pl.DataFrame({
        'X': points_np[:, 0],
        'Y': points_np[:, 1],
        'Z': points_np[:, 2],
        'R_norm': colors_np[:, 0],
        'G_norm': colors_np[:, 1],
        'B_norm': colors_np[:, 2]
    })

        df_pcd = df_pcd.with_columns(
            R = (pl.col('R_norm') * 255).cast(pl.Int32),
            G = (pl.col('G_norm') * 255).cast(pl.Int32),
            B = (pl.col('B_norm') * 255).cast(pl.Int32)
        ).with_columns(
            color_hex = (
                pl.lit('rgb(') + pl.col('R').cast(pl.String) + pl.lit(',') +
                pl.col('G').cast(pl.String) + pl.lit(',') +
                pl.col('B').cast(pl.String) + pl.lit(')')
            )
        )

        df_plotly = df_pcd.to_pandas()



        fig = px.scatter_3d(
            df_plotly, 
            x='X',
            y='Y',
            z='Z',
            color='color_hex',  
            color_discrete_map="identity", 
            height=600,
            title="Interactive Open3D Point Cloud in Marimo (via Polars/Plotly)",
            hover_data=['R', 'G', 'B']
        )

        fig.update_traces(marker=dict(size=2)) 
        fig.update_layout(scene=dict(aspectmode='data')) 

        return mo.ui.plotly(fig)
        
        
    return (draw_3d_model,)


@app.cell
def _(asuka_3d, draw_3d_model):
    draw_3d_model(asuka_3d)
    return


@app.cell
def _(asuka_best_reconstruction):
    asuka_best_reconstruction
    return


@app.cell
def _(asuka_best_reconstruction):
    asuka_best_reconstruction.frames[191].has_pose()
    return


@app.cell
def _(asuka_best_reconstruction):
    points, cameras, images = asuka_best_reconstruction.points3D ,asuka_best_reconstruction.cameras, asuka_best_reconstruction.images
    return (images,)


@app.cell
def _(images):
    images[191].points2D[0]
    return


@app.cell
def _(
    Path,
    defaultdict,
    images,
    load_sam2_masks,
    pixels_to_mask,
    pycolmap,
    tqdm,
):
    def get_segmented_3D_points(
        reconstruction_path: Path,
        images_path: Path,
        masks_dir: Path,
        min_track_length: int = 3,
        min_mask_ratio: float = 0.5,
    ) -> pycolmap.Reconstruction:

        reconstruction = pycolmap.Reconstruction(reconstruction_path)
        segmented_reconstruction = pycolmap.Reconstruction(reconstruction_path)

        points3D = reconstruction.points3D
        valid_points = set()
        mask_hit_counts = defaultdict(int)

        for image_idx, image in tqdm(images.items(), desc="Processing images"):

            image_path = images_path / "png_frames" / Path(image.name).name
            masks_json_path = masks_dir / f"{Path(image.name).stem}_masks.json"

            masks_data = load_sam2_masks(masks_json_path)
            objects = masks_data['objects']
            for relevant_object in objects:
                if relevant_object["object_id"] == 1:
                    break
            mask = pixels_to_mask(
                relevant_object['pixels'],
                masks_data['image_size']['height'],
                masks_data['image_size']['width']
            )
        
            for point2D in image.points2D:
                if point2D.has_point3D():
                    x, y = point2D.xy
                    x, y = int(round(x)), int(round(y))
                    if mask[x,y] == 1:
                        mask_hit_counts[point2D.point3D_id] += 1

        points_to_delete = []
        for point3D_id, point3D in reconstruction.points3D.items():
        
            total_observations = point3D.track.length()
        
            mask_hits = mask_hit_counts[point3D_id]
        
            ratio = mask_hits / total_observations if total_observations > 0 else 0
        
            if total_observations < min_track_length or ratio < min_mask_ratio:
                points_to_delete.append(point3D_id)

        for point3D_id in points_to_delete:
            segmented_reconstruction.delete_point3D(point3D_id)
                
        return segmented_reconstruction
    return (get_segmented_3D_points,)


@app.cell
def _(Path, get_segmented_3D_points, images_path):
    segmented_reconstruction = get_segmented_3D_points(
        reconstruction_path=Path("./reconstruction/0"),
        images_path=images_path,
        masks_dir=Path("./data/sam2_masks")
    )
    return (segmented_reconstruction,)


@app.cell
def _(reconstruction_path, segmented_reconstruction):
    segmented_reconstruction.export_PLY(reconstruction_path / "segmented_model.ply")
    segmented_reconstruction.write("./reconstruction/segmented_reconstruction")
    return


@app.cell
def _(o3d, reconstruction_path):
    segmented_reconstruction_o3d = o3d.io.read_point_cloud(str(reconstruction_path / "segmented_model.ply"))
    cl, ind = segmented_reconstruction_o3d.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
    clean_segmented_reconstruction_o3d = segmented_reconstruction_o3d.select_by_index(ind)
    return (clean_segmented_reconstruction_o3d,)


@app.cell
def _(clean_segmented_reconstruction_o3d, draw_3d_model):
    draw_3d_model(clean_segmented_reconstruction_o3d)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
