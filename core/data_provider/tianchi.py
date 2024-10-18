import os
import cv2
import numpy as np
import random
import rarfile

class InputHandle:
    def __init__(self, input_param):
        self.paths = input_param['paths']
        self.num_paths = len(input_param['paths'])
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.output_data_type = input_param.get('output_data_type', 'float32')
        self.minibatch_size = input_param['minibatch_size']
        self.is_output_sequence = input_param['is_output_sequence']
        self.target_size = (128, 128)  # 目标尺寸 128x128
        self.data = []
        self.indices = []
        self.current_position = 0
        self.current_batch_size = 0
        self.current_batch_indices = []
        self.load()

    def load(self):
        """Load data from the rar files."""
        for path in self.paths:
            with rarfile.RarFile(path) as rar:
                # 获取所有文件夹列表
                for folder_name in rar.namelist():
                    # 确保是 sample_x 文件夹
                    if folder_name.endswith('/') and 'sample_' in folder_name:
                        sample_images = []
                        # 每个文件夹有15张图片，编号从1到15
                        for i in range(1, 16):
                            img_name = f"{folder_name}img_{i}.png"
                            try:
                                # 从rar文件中读取图片
                                with rar.open(img_name) as img_file:
                                    # 解码并读取图片 (img_channel, img_height, img_width)
                                    img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_COLOR)
                                    # 调整图片大小为128x128
                                    img_resized = cv2.resize(img, self.target_size)
                                    # 转换为 (img_height, img_width, img_channel)
                                    sample_images.append(img_resized)
                            except Exception as e:
                                print(f"Error reading {img_name}: {e}")
                        # 确保加载了15张图片
                        if len(sample_images) == 15:
                            self.data.append(sample_images)
                        else:
                            print(f"Skipping folder {folder_name} due to incomplete images.")

        print(f"Loaded {len(self.data)} samples from {self.paths}")
        self.indices = np.arange(len(self.data), dtype="int32")

    def total(self):
        return len(self.data)

    def begin(self, do_shuffle=True):
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        self.current_batch_size = min(self.minibatch_size, self.total())
        self.current_batch_indices = self.indices[:self.current_batch_size]

    def next(self):
        self.current_position += self.current_batch_size
        if self.no_batch_left():
            return None
        self.current_batch_size = min(self.minibatch_size, self.total() - self.current_position)
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.current_batch_size]

    def no_batch_left(self):
        return self.current_position >= self.total() - self.current_batch_size

    def input_batch(self):
        input_batch = []
        for ind in self.current_batch_indices:
            sample = self.data[ind]
            inputs = np.array(sample[:5], dtype=self.input_data_type)  # 选择前5张图片作为输入
            input_batch.append(inputs)
        input_batch = np.array(input_batch)
        return input_batch

    def output_batch(self):
        output_batch = []
        for ind in self.current_batch_indices:
            sample = self.data[ind]
            outputs = np.array(sample[5:], dtype=self.output_data_type)  # 后10张作为输出
            output_batch.append(outputs)
        output_batch = np.array(output_batch)
        return output_batch

    def get_batch(self):
        input_seq = self.input_batch()
        output_seq = self.output_batch()
        batch = np.concatenate((input_seq, output_seq), axis=1)
        return batch




if __name__ == "__main__":
    from test_util.visualize_batch import visualize_batch
    # 定义输入参数
    input_param = {
        'paths': ['/home/huangzhe/PrenRNN/data/tianchi-example/train.rar'],  # 修改为实际文件路径
        'name': 'tianchi_dataset',
        'minibatch_size': 8,  # 每个批次大小
        'is_output_sequence': True,
        'input_data_type': 'float32',
        'output_data_type': 'float32'
    }

    # 实例化 InputHandle 类
    input_handle = InputHandle(input_param)

    # 开始新的批次处理
    input_handle.begin(do_shuffle=True)

    # 获取一个批次数据
    i=0
    batch = input_handle.get_batch()
    print(np.max(batch))
    print(batch.dtype)


    # 打印或处理批次数据
    print(f"Batch shape: {batch.shape}")
    # 输入 (batch_size, seq_length, img_height, img_width, img_channel)
    print(f"Input batch sample: {batch[0, 0].shape}")

