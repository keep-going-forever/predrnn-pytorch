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
            print("区域MSE计算")
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

    pod_per_frame = []
    far_per_frame = []
    csi_per_frame = []
    hss_per_frame = []
    mae_per_frame = []  # To store MAE for each frame

    for i in range(configs.total_length - configs.input_length):
        img_mse.append(0)
        ssim.append(0)
        psnr.append(0)
        lp.append(0)

        pod_per_frame.append(0)
        far_per_frame.append(0)
        csi_per_frame.append(0)
        hss_per_frame.append(0)
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

    while (test_input_handle.no_batch_left() == False):
        batch_id = batch_id + 1
        test_ims = test_input_handle.get_batch()
        if(np.max(test_ims)>1):
            test_ims = test_ims / 255.0
        test_dat = preprocess.reshape_patch(test_ims, configs.patch_size)
        test_ims = test_ims[:, :, :, :, :configs.img_channel]
        img_gen = model.test(test_dat, real_input_flag)

        img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
        output_length = configs.total_length - configs.input_length
        img_out = img_gen[:, -output_length:]

        # MSE and MAE Calculation and Metrics for each frame
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

            # Binary classification for POD, FAR, CSI, and HSS
            threshold = configs.binary_threshold  # Define a threshold for binary classification
            x_binary = (x >= threshold).astype(int)
            gx_binary = (gx >= threshold).astype(int)

            # Loop through each batch and calculate confusion matrix
            for b in range(configs.batch_size):
                tn, fp, fn, tp = confusion_matrix(
                    x_binary[b].flatten(), gx_binary[b].flatten(), labels=[0, 1]
                ).ravel()

                # Calculate POD, FAR, CSI, and HSS
                pod = tp / (tp + fn) if (tp + fn) > 0 else 0
                far = fp / (tp + fp) if (tp + fp) > 0 else 0
                csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
                hss = (
                    2 * (tp * tn - fp * fn)
                    / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
                    if ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)) > 0
                    else 0
                )

                # Accumulate per frame results
                pod_per_frame[i] += pod
                far_per_frame[i] += far
                csi_per_frame[i] += csi
                hss_per_frame[i] += hss

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
    print('mse per seq:', avg_mse)
    print('mae per seq:', avg_mae)

    for i in range(configs.total_length - configs.input_length):
        print(f'MSE per frame {i + 1}:', img_mse[i] / (batch_id * configs.batch_size))
        print(f'MAE per frame {i + 1}:', mae_per_frame[i] / (batch_id * configs.batch_size))

    # Display POD, FAR, CSI, HSS per frame
    print('POD per frame:')
    for i in range(configs.total_length - configs.input_length):
        print(f'Frame {i + 1}:', pod_per_frame[i] / batch_id)

    print('FAR per frame:')
    for i in range(configs.total_length - configs.input_length):
        print(f'Frame {i + 1}:', far_per_frame[i] / batch_id)

    print('CSI per frame:')
    for i in range(configs.total_length - configs.input_length):
        print(f'Frame {i + 1}:', csi_per_frame[i] / batch_id)

    print('HSS per frame:')
    for i in range(configs.total_length - configs.input_length):
        print(f'Frame {i + 1}:', hss_per_frame[i] / batch_id)





