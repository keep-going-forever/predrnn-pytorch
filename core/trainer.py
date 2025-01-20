import os.path
import datetime
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from core.utils import preprocess, metrics
import torch
from test_util.test_util import save_radar_images
from sklearn.metrics import confusion_matrix




def train(model, ims, real_input_flag, configs, itr):
    if configs.is_regional:
        try:
            # print("区域MSE计算")
            cost = model.regional_train(ims, real_input_flag)
        except Exception as e:
            print(f"Error in regional_train: {e}")
            raise  # 可以根据实际情况决定是直接抛出异常还是进行其他补救操作
    else:
        try:
            cost = model.train(ims, real_input_flag)
        except Exception as e:
            print(f"Error in train: {e}")
            raise
    if configs.reverse_input:
        ims_rev = np.flip(ims, axis=1).copy()
        cost += model.train(ims_rev, real_input_flag)
        cost = cost / 2

    if itr % configs.display_interval == 0:
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'itr: ' + str(itr))
        print('training loss: ' + str(cost))





def calculate_csi(predicted, true, threshold_min, threshold_max):
    """
    计算给定阈值范围内的CSI值。

    参数:
    - predicted: 预测值的数组，形状为[batch, h, w, c]
    - true: 真实值的数组，形状为[batch, h, w, c]
    - threshold_min: 阈值的最小值
    - threshold_max: 阈值的最大值

    返回:
    - csi: CSI值
    """
    # 确保输入的维度相同
    assert predicted.shape == true.shape, "预测值和真实值的形状必须相同"
    predicted = predicted*70
    true = true*70

    # 扩展阈值范围
    pred_within_threshold = (predicted >= threshold_min) & (predicted < threshold_max)
    true_within_threshold = (true >= threshold_min) & (true < threshold_max)

    # 计算命中、误报和漏报
    hit = np.sum(pred_within_threshold & true_within_threshold)
    false_alarm = np.sum(pred_within_threshold & ~true_within_threshold)
    miss = np.sum(~pred_within_threshold & true_within_threshold)

    # 避免分母为零
    denominator = hit + false_alarm + miss
    if denominator == 0:
        csi = 0  # 或者其他表示无效值的标志
    else:
        csi = hit / denominator
    return csi




def test(model, test_input_handle, configs, itr):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'test...')
    test_input_handle.begin(do_shuffle=False)
    res_path = os.path.join(configs.gen_frm_dir, str(itr))
    os.mkdir(res_path)
    avg_mse = 0
    avg_mae = 0
    batch_id = 0
    img_mse, ssim, psnr = [], [], []
    lp = []

    mae_per_frame = []  # To store MAE for each frame

    for i in range(configs.total_length - configs.input_length):
        img_mse.append(0)
        ssim.append(0)
        psnr.append(0)
        lp.append(0)
        mae_per_frame.append(0)  # Initialize MAE for each frame

    # reverse schedule sampling
    if configs.reverse_scheduled_sampling == 1:
        mask_input = 1
    else:
        mask_input = configs.input_length

    real_input_flag = np.zeros(
        (configs.batch_size,
         configs.total_length - mask_input - 1,
         configs.img_width // configs.patch_size,
         configs.img_width // configs.patch_size,
         configs.patch_size ** 2 * configs.img_channel))

    if configs.reverse_scheduled_sampling == 1:
        real_input_flag[:, :configs.input_length - 1, :, :] = 1.0

    csi_20_30_total = 0
    csi_30_40_total = 0
    csi_above_40_total = 0

    while not test_input_handle.no_batch_left():
        batch_id = batch_id + 1
        test_ims = test_input_handle.get_batch()



        if np.max(test_ims) > 1:
            test_ims = test_ims / 255.0

        # 统计数据分布
        test_ims_scaled = test_ims * 70
        count_0_10 = np.sum((test_ims_scaled >= 0) & (test_ims_scaled < 10))
        count_10_20 = np.sum((test_ims_scaled >= 10) & (test_ims_scaled < 20))
        count_20_30 = np.sum((test_ims_scaled >= 20) & (test_ims_scaled < 30))
        count_30_40 = np.sum((test_ims_scaled >= 30) & (test_ims_scaled < 40))
        count_above_40 = np.sum(test_ims_scaled >= 40)


        test_dat = preprocess.reshape_patch(test_ims, configs.patch_size)
        test_ims = test_ims[:, :, :, :, :configs.img_channel]
        img_gen = model.test(test_dat, real_input_flag)

        img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
        output_length = configs.total_length - configs.input_length
        img_out = img_gen[:, -output_length:]

        # MSE, MAE, and CSI Calculation and Metrics for each frame
        for i in range(output_length):
            x = test_ims[:, i + configs.input_length, :, :, :]
            gx = img_out[:, i, :, :, :]
            gx = np.maximum(gx, 0)
            gx = np.minimum(gx, 1)
            mse = np.square(x - gx).sum()
            img_mse[i] += mse
            avg_mse += mse

            # MAE Calculation
            mae = np.abs(x - gx).sum()
            mae_per_frame[i] += mae
            avg_mae += mae

            # CSI Calculation for current frame
            csi_20_30 = calculate_csi(gx, x, 20, 30)
            csi_30_40 = calculate_csi(gx, x, 30, 40)
            csi_above_40 = calculate_csi(gx, x, 40, np.inf)
            csi_20_30_total += csi_20_30
            csi_30_40_total += csi_30_40
            csi_above_40_total += csi_above_40

        # Save prediction examples
        if batch_id <= configs.num_save_samples:
            path = os.path.join(res_path, str(batch_id))
            os.mkdir(path)
            for i in range(configs.total_length):
                name = f'gt{i + 1}.png'
                file_name = os.path.join(path, name)
                img_gt = np.uint8(test_ims[0, i, :, :, :] * 255)
                cv2.imwrite(file_name, img_gt)
            for i in range(output_length):
                name = f'pd{i + 1 + configs.input_length}.png'
                file_name = os.path.join(path, name)
                img_pd = img_out[0, i, :, :, :]
                img_pd = np.maximum(img_pd, 0)
                img_pd = np.minimum(img_pd, 1)
                img_pd = np.uint8(img_pd * 255)
                cv2.imwrite(file_name, img_pd)

        test_input_handle.next()

    avg_mse /= (batch_id * configs.batch_size)
    avg_mae /= (batch_id * configs.batch_size)
    csi_20_30_total /= (batch_id * output_length)
    csi_30_40_total /= (batch_id * output_length)
    csi_above_40_total /= (batch_id * output_length)

    print('mse per seq:', avg_mse)
    print('mae per seq:', avg_mae)

    for i in range(configs.total_length - configs.input_length):
        print(f'MSE per frame {i + 1}:', img_mse[i] / (batch_id * configs.batch_size))
    for i in range(configs.total_length - configs.input_length):
        print(f'MAE per frame {i + 1}:', mae_per_frame[i] / (batch_id * configs.batch_size))

    print(f'CSI (20-30): {csi_20_30_total}')
    print(f'CSI (30-40): {csi_30_40_total}')
    print(f'CSI (>40): {csi_above_40_total}')







