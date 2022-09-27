from torch.utils.data import DataLoader
from dataset.dataset_wifi import WifiDataset


def get_dataloader(phase, config, is_shuffle=None):
    is_shuffle = phase == 'train' if is_shuffle is None else is_shuffle

    if config.module == 'wifi':
        dataset = WifiDataset(phase, config)
    else:
        raise NotImplementedError
    
    # for i in range(len(dataset)):
    #     print(i)
    #     dataset[i]
    # exit()
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle,
                                num_workers=config.num_workers)
    return dataloader
