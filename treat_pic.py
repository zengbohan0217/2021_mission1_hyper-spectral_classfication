import CNN_FC_model as CNN_F
import torch
import numpy as np
from tqdm import tqdm

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

class hyper_spectral():
    def __init__(self, input_pic):
        """
        :param input_pic: a hyper_spectral_pic array
        """
        self.hyper_pic = input_pic
        self.pic_C = len(input_pic)
        self.pic_H = len(input_pic[0])
        self.pic_W = len(input_pic[0][0])

    def cut_off(self):
        # cut the original pic into several channel*3*3 blocks
        out_put = []
        for i in range(0, self.pic_H+1, 3):
            for j in range(0, self.pic_W + 1, 3):
                if i+3 < self.pic_H+1 and j+3 < self.pic_W+1:
                    c = self.hyper_pic[0:self.pic_C+1, i:i+3, j:j+3]
                    out_put.append(c)
        out_put = np.array(out_put)
        return out_put

    def part_classification(self, batch_size):
        part_list = self.cut_off()
        print(part_list.shape)
        model = CNN_F.CNN3Net_224().to(DEVICE)
        # model = CNN_F.CNN3Net_102().to(DEVICE)
        out_list = []
        for i in tqdm(range(0, len(part_list)+1, batch_size)):
            if i+batch_size < len(part_list)+1:
                c = torch.tensor(part_list[i:i+batch_size], dtype=torch.float32).to(DEVICE)
                out = model(c)
                out_list.append(out)
        final_list = out_list.pop(0)
        for out in out_list:
            final_list = torch.cat((final_list, out))
        return final_list



