## Project Notes

One can find the source video, extracted frames, SAM2 masks, and COLMAP reconstructions [under this link](https://drive.google.com/drive/folders/1fe27d71lrdMEif8RSsaaTU1_XB1Gm91N?usp=sharing).

## Why This Object?

- **Subject**: A brightly colored anime figurine (Asuka) placed on a plain table. The figurine has high-frequency textures (hair, suit decals) and lots of non-coplanar geometry, which helps COLMAP extract stable SIFT features and recover a dense point cloud.
- **Lighting**: Soft indoor lighting reduces harsh specular highlights, so SAM2 and COLMAP both receive consistent color information.
Background surface: The table is smooth, largely Lambertian, and has low texture, which resulted in a sparse point cloud reconstruction of the surface. While standard RANSAC plane fitting is effective for dense surfaces, the low point density here made it unreliable. Consequently, the implementation utilizes a PCA-based separation method: we identify the object's primary axis (eigenvector) and remove the background by statistically clipping the extreme percentile of points projected along that vector, effectively slicing off the noise without requiring a dense plane model.
- **Geometry influence**: The figurine’s thin limbs introduce occlusions, so keeping a tight frame interval (every other frame) was important to ensure COLMAP had overlapping features from multiple viewpoints. Because the object has vivid color changes, per-pixel SAM2 masks remain stable even when motion blur appears between frames.

## Foreground vs. Background Separation

### Mask-Based Filtering (2D → 3D projection)
- **Pros**: Highly precise when the object mask is accurate; works even when the background is not planar; leverages temporal consistency because each 3D point must be visible in multiple labeled images.
- **Cons**: Requires running SAM2 (or another segmenter) on every frame and ensuring filenames stay synced with COLMAP images; gaps or mis-segmented frames can remove good points or keep noise.

### Geometric Plane Removal (Open3D RANSAC / PCA)
- **Pros**: Fast, operates purely in 3D, no dependency on per-frame masks once the cloud is built; ideal when the background is a dominant plane (table, floor).
- **Cons**: Fails if the background is not planar or when the object itself includes large planar regions parallel to the table; sensitive to RANSAC thresholds and may require manual tuning to avoid removing lower parts of the object.

### Hybrid Strategy (Used Here)
- Start with SAM2-guided filtering to prune obvious non-object points before converting to Open3D.
- Apply RANSAC/PCA-based plane removal on the cleaned cloud to drop residual table points and outliers.
- **Benefit**: Combines the semantic accuracy of 2D masks with the robustness of 3D geometry filters, producing a compact, high-quality point cloud of the figurine.
