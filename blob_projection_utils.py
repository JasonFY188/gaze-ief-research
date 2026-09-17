import numpy as np
import cv2
import os


def project_point_net(P, R, t, K, H_net, W_net):
    """
    Project a 3D point P (world) into network space (H_net, W_net).
    Returns (u_net, v_net) or None if behind camera or outside FOV.
    """

    # Transform P into camera coordinates
    Pc = R @ P + t

    # Behind camera
    if Pc[2] <= 1e-6:
        return None

    # Intrinsic projection
    x = K[0, 0] * (Pc[0] / Pc[2]) + K[0, 2]
    y = K[1, 1] * (Pc[1] / Pc[2]) + K[1, 2]

    # Check range in network coordinates
    if 0 <= x < W_net and 0 <= y < H_net:
        return x, y
    return None



def project_blob_to_images(
    blob_points,
    orig_images,
    orig_sizes,
    extrinsic_np,
    intrinsic_np,
    H_net,
    W_net,
    mask_method="kde",   # "hull" | "morph" | "kde"
):
    """
    Project 3D blob_points into original images and build 2D masks.
    Returns:
        images_bgr: list of (H_i, W_i, 3)
        masks:      list of (H_i, W_i)
    """

    S = len(orig_images)
    images_bgr = []
    masks = []

    for i in range(S):
        # Original RGB → BGR
        img_rgb = orig_images[i]
        img = img_rgb[:, :, ::-1].copy()
        H_i, W_i = orig_sizes[i]

        R = extrinsic_np[i, :, :3]
        t = extrinsic_np[i, :, 3]
        K = intrinsic_np[i]

        dot_list = []

        # ======================================================
        # 1. Project each 3D point into this view
        # ======================================================
        for P in blob_points:
            uv_net = project_point_net(P, R, t, K, H_net, W_net)
            if uv_net is None:
                continue

            u_net, v_net = uv_net

            # Scale from VGGT net to original image
            u_orig = u_net * (W_i / W_net)
            v_orig = v_net * (H_i / H_net)

            if 0 <= u_orig < W_i and 0 <= v_orig < H_i:
                # Round → clip to valid range
                u_i = int(np.clip(round(u_orig), 0, W_i - 1))
                v_i = int(np.clip(round(v_orig), 0, H_i - 1))

                # Draw dot
                cv2.circle(img, (u_i, v_i), 3, (0, 0, 255), -1)

                dot_list.append((u_i, v_i))



        # ======================================================
        # 2. Construct mask from dot_list
        # ======================================================
        mask = np.zeros((H_i, W_i), dtype=np.uint8)

        if len(dot_list) > 0:

            # ---------------- CONVEX HULL ----------------
            if mask_method == "hull":
                if len(dot_list) > 3:
                    pts = np.array(dot_list, dtype=np.int32)
                    hull = cv2.convexHull(pts)
                    cv2.fillConvexPoly(mask, hull, 255)
                else:
                    for (u, v) in dot_list:
                        mask[v, u] = 255


            # ---------------- MORPHOLOGICAL BLOB ----------------
            elif mask_method == "morph":
                for (u, v) in dot_list:
                    mask[v, u] = 255

                kernel = np.ones((15, 15), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=3)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


            # ---------------- KDE HEATMAP MASK ----------------
            elif mask_method == "kde":
                heat = np.zeros((H_i, W_i), dtype=np.float32)

                for (u, v) in dot_list:
                    cv2.circle(heat, (u, v), 8, 1.0, -1)

                # Smooth strongly
                heat = cv2.GaussianBlur(heat, (45, 45), 8)

                if heat.max() > 0:
                    heat /= heat.max()

                mask = (heat > 0.15).astype(np.uint8) * 255

                # Fix holes & noise
                kernel = np.ones((15, 15), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

                # Keep only largest connected component
                num_labels, labels = cv2.connectedComponents(mask)
                if num_labels > 1:
                    sizes = [(labels == j).sum() for j in range(1, num_labels)]
                    best = 1 + np.argmax(sizes)
                    mask = (labels == best).astype(np.uint8) * 255

                # Smooth edges a bit
                mask = cv2.dilate(mask, np.ones((13, 13), np.uint8), iterations=1)



        # ======================================================
        # 3. Final smoothing
        # ======================================================
        if np.any(mask):
            mask = cv2.GaussianBlur(mask, (9, 9), 0)
            mask = (mask > 20).astype(np.uint8) * 255

        # Append results
        images_bgr.append(img)
        masks.append(mask)

    return images_bgr, masks




