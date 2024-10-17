import matplotlib.pyplot as plt
import numpy as np
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