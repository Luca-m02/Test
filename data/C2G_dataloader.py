import torch


class C2GDataloader():
    def __init__(self, args, ds_parser):
        self.args = args
        self.batch_size = args.batch_size
        self.train_dataset, self.val_dataset, self.test_dataset, self.x_min, self.x_max, self.y_min, self.y_max = ds_parser.get_datasets()

    def get_dataloaders(self):
        train_loader = self._get_loader(self.train_dataset, shuffle=True, drop_last=True)
        val_loader = self._get_loader(self.val_dataset, shuffle=False, drop_last=False)
        test_loader = self._get_loader(self.test_dataset, shuffle=False, drop_last=False)
        return train_loader, val_loader, test_loader, self.x_min, self.x_max, self.y_min, self.y_max

    def _get_loader(self, dataset, shuffle, drop_last, num_workers=1):
        dataloader = torch.utils.data.DataLoader(dataset,
                                           batch_size=self.batch_size,
                                           shuffle=shuffle)
        return dataloader
