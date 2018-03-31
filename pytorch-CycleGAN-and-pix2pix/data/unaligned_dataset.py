import os.path
from data.base_dataset import BaseDataset, get_transform, toTensor_normalize
from data.image_folder import make_dataset
from PIL import Image
import random
import pdb


class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt)
        self.toTensor_normalize = toTensor_normalize(opt)

    def get_map_path(self, trainPath, A_or_B):
        trainPath = os.path.normpath(trainPath)
        map_path = trainPath.split(os.sep)
        map_path[-2] = 'mask' + A_or_B
        return os.path.join(*map_path)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        #A = self.transform(A_img)
        #B = self.transform(B_img)

        A = self.toTensor_normalize(A_img)
        B = self.toTensor_normalize(B_img)

        # make call to semantic segmentation for producing A_mask
        if self.opt.lambda_mask > 0.0:
            # pdb.set_trace()
            A_map_path = self.get_map_path(A_path, 'A')
            B_map_path = self.get_map_path(B_path, 'B')
            A_mask = self.toTensor_normalize(Image.open(A_map_path))
            B_mask = self.toTensor_normalize(Image.open(B_map_path))

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        if self.opt.lambda_mask > 0.0:
            return {'A': A, 'A_mask': A_mask, 'B': B, 'B_mask': B_mask,
                    'A_paths': A_path, 'B_paths': B_path}

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
