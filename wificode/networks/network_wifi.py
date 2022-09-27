import torch
import torch.nn as nn

class WifiNet(nn.Module):
    def __init__(self, cfg):
        super(WifiNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1070, 2048),
            #nn.LeakyReLU(inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            #nn.LeakyReLU(inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            #nn.LeakyReLU(inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1))
        self.softmax = nn.Softmax(dim=1)
                       

    def forward(self, src_locs, rssi_pairs):
        batch_size, K, _ = rssi_pairs.size()
        rssi_pairs = rssi_pairs.view(batch_size*K, rssi_pairs.size(2))
        rssi_pairs = self.encoder(rssi_pairs)
        rssi_pairs = rssi_pairs.view(batch_size, K, 1)
        #dist = rssi_pairs
        rssi_pairs = self.softmax(rssi_pairs)
        src_locs = rssi_pairs*src_locs
        src_locs = torch.sum(src_locs, 1, keepdim=True)
        return src_locs#, dist

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input_channel', type=int, default=7, help='')
    parser.add_argument('--arch', type=str, default='resnet50', help='')
    parser.add_argument('--pretrained', type=bool, default=False, help='')

    args = parser.parse_args()

    net = Baseline(args)
    h, w = 256, 384
    a = torch.zeros(16,3, h,w)
    b = torch.zeros(16,3, h,w)
    c = torch.zeros(16,1, h,w)
    d = net(a,b,c)
    print(d.size())