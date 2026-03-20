import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image

# Point-guided image deformation
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    Return
    ------
        A deformed image.
    """

    warped_image = np.array(image)
    # 1. 基础检查与变量准备
    if len(source_pts) < 2 or len(source_pts) != len(target_pts):
        return np.array(image)

    h, w, _ = image.shape
    alpha = 1.0
    eps = 1e-8

    # 2. 生成目标图像的像素网格
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    v = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2).astype(np.float32)  # (N, 2)

    # 3. 计算权重,计算v到target_pts的距离
    # 因为我们要反向查找：目标图的点 v 对应原图哪个 p
    # target_pts: (n, 2), v: (N, 2) -> diff: (n, N, 2)
    diff = target_pts[:, np.newaxis, :] - v[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2) + eps
    weights = 1.0 / (dist_sq ** alpha)  # (n, N)
    weights /= np.sum(weights, axis=0)  # 归一化 (n, N)

    # 4. 计算加权质心
    # q_star 是目标图(当前网格)的质心，p_star 是原图(来源)的质心
    q_star = np.sum(weights[:, :, np.newaxis] * target_pts[:, np.newaxis, :], axis=0)  # (N, 2)
    p_star = np.sum(weights[:, :, np.newaxis] * source_pts[:, np.newaxis, :], axis=0)  # (N, 2)

    # 5. 中心化坐标
    q_hat = target_pts[:, np.newaxis, :] - q_star  # (n, N, 2)
    p_hat = source_pts[:, np.newaxis, :] - p_star  # (n, N, 2)
    v_minus_qstar = v - q_star  # (N, 2)

    # 6. 计算相似性变换 (Similarity Transform)
    # 构造垂直向量
    q_hat_perp = np.stack([-q_hat[:, :, 1], q_hat[:, :, 0]], axis=-1)
    v_minus_qstar_perp = np.stack([-v_minus_qstar[:, 1], v_minus_qstar[:, 0]], axis=-1)

    # 分母 mu
    mu = np.sum(weights * np.sum(q_hat ** 2, axis=2), axis=0)  # (N,)

    # 计算映射后的坐标 f(v) -> 映射回原图的位置
    f_v = np.zeros_like(v)
    for i in range(len(source_pts)):
        s_i = (weights[i] / mu) * np.sum(q_hat[i] * v_minus_qstar, axis=1)
        h_i = (weights[i] / mu) * np.sum(q_hat_perp[i] * v_minus_qstar, axis=1)

        p_i_perp = np.stack([-p_hat[i, :, 1], p_hat[i, :, 0]], axis=-1)
        f_v += (s_i[:, np.newaxis] * p_hat[i] + h_i[:, np.newaxis] * p_i_perp)

    f_v += p_star  # 得到目标点 v 在原图对应的坐标

    # 7. 重采样渲染
    map_x = f_v[:, 0].reshape(h, w).astype(np.float32)
    map_y = f_v[:, 1].reshape(h, w).astype(np.float32)

    # 限制范围防止越界
    map_x = np.clip(map_x, 0, w - 1)
    map_y = np.clip(map_y, 0, h - 1)

    warped_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    return warped_image

def run_warping():
    global points_src, points_dst, image

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()