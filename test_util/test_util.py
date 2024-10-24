import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

def save_radar_images(test_ims, img_out, batch_id, res_path, configs, max_dbz=80):
    """
    将雷达回波图从归一化的 [0, 1] 反归一化到 dBZ 范围，并根据 dBZ 值着色保存图片。

    参数：
    - test_ims: 真实图像数据，形状为 (batch_size, total_length, height, width, channels)。
    - img_out: 预测图像数据，形状为 (batch_size, output_length, height, width, channels)。
    - batch_id: 当前批次 ID。
    - res_path: 保存图片的路径。
    - configs: 配置对象，包含 total_length, input_length 和 num_save_samples 等参数。
    - max_dbz: dBZ 最大阈值，用于反归一化。
    """
    # 确保保存路径存在
    res_path = os.path.join(res_path, 'radar')
    if batch_id <= configs.num_save_samples:
        path = os.path.join(res_path, str(batch_id))
        os.makedirs(path, exist_ok=True)

        # 颜色映射表，基于雷达回波反射率 (dBZ)
        color_map = [
            (0, 0, 255),    # 深蓝色，0-10 dBZ
            (0, 128, 255),  # 浅蓝色，10-20 dBZ
            (0, 255, 0),    # 绿色，20-30 dBZ
            (255, 255, 0),  # 黄色，30-40 dBZ
            (255, 165, 0),  # 橙色，40-50 dBZ
            (255, 0, 0),    # 红色，50-60 dBZ
            (139, 0, 0),    # 深红色，60-70 dBZ
            (128, 0, 128)   # 紫色，> 70 dBZ
        ]

        # 定义反射率的区间范围
        dbz_thresholds = [10, 20, 30, 40, 50, 60, 70, max_dbz]

        # 保存真实图像
        for i in range(configs.total_length):
            name = 'gt' + str(i + 1) + '.png'
            file_name = os.path.join(path, name)

            # 将归一化的图像反归一化到 dBZ 范围
            img_gt = test_ims[0, i, :, :, :] * max_dbz


            # 将 dBZ 图像转换为彩色图像
            img_color_gt = dbz_to_color(img_gt, dbz_thresholds, color_map)

            # 保存图像
            cv2.imwrite(file_name, img_color_gt)

        # 保存预测图像
        for i in range(img_out.shape[1]):
            name = 'pd' + str(i + 1 + configs.input_length) + '.png'
            file_name = os.path.join(path, name)

            # 将归一化的预测图像反归一化到 dBZ 范围
            img_pd = img_out[0, i, :, :, :] * max_dbz
            img_pd = np.clip(img_pd, 0, max_dbz)

            # 将 dBZ 图像转换为彩色图像
            img_color_pd = dbz_to_color(img_pd, dbz_thresholds, color_map)

            # 保存图像
            cv2.imwrite(file_name, img_color_pd)


def dbz_to_color(dbz_img, thresholds, colors):
    """
    将 dBZ 图像转换为彩色图像。

    参数：
    - dbz_img: dBZ 图像，形状为 (height, width, 1)。
    - thresholds: dBZ 阈值列表。
    - colors: 颜色映射表，与 dBZ 阈值对应。

    返回值：
    - 彩色图像，形状为 (height, width, 3)。
    """
    height, width, _ = dbz_img.shape
    img_color = np.zeros((height, width, 3), dtype=np.uint8)

    # 将 dbz_img 从 (height, width, 1) 转换为 (height, width)
    dbz_img_2d = dbz_img[:, :, 0]

    # 遍历每个阈值区间，根据 dBZ 值分配颜色
    for i, threshold in enumerate(thresholds):
        # 创建布尔掩码
        lower_bound = thresholds[i-1] if i > 0 else 0
        mask = (dbz_img_2d >= lower_bound) & (dbz_img_2d < threshold)

        # 使用 mask 选择位置并分配颜色
        img_color[mask] = colors[i]

    # 处理大于 max_dbz 的情况
    mask = dbz_img_2d >= thresholds[-1]
    img_color[mask] = colors[-1]

    return img_color



def visualize_batch(batch):
    batch_size, seq_length, img_height, img_width, img_channel = batch.shape

    # 遍历 batch 中的每一个样本
    for i in range(batch_size):
        print(f"Visualizing sample {i + 1}/{batch_size}")

        # 为每个样本创建一个图形
        fig, axs = plt.subplots(1, seq_length, figsize=(20, 5))

        for j in range(seq_length):
            img = batch[i, j]  # 取得图片 (128, 128, 3)
            axs[j].imshow(img.astype(np.uint8))  # 显示图片
            axs[j].axis('off')  # 关闭坐标轴
            axs[j].set_title(f"img_{j + 1}")

        # 显示当前样本的图片
        plt.show()