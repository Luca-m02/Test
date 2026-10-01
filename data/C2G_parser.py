import os
import numpy            as     np
import pandas           as     pd
import logging
from torch.utils.data import DataLoader, TensorDataset
import torch

class C2G_parser:
    def __init__(self, args):
        self.data_location = os.path.join(args.data_path)
        self.house_indicies_train = args.house_indicies_train
        self.house_indicies_test = args.house_indicies_test
        self.key = 'timestamp'
        self.sep = ';'
        self.fuso = args.fuso

        self.train_size = args.train_size
        self.window_len = args.window_len
        self.window_stride = args.window_stride
        self.resample_period = args.resample_period

        self.x_train, self.y_train, self.x_val, self.y_val = self.load_data(self.house_indicies_train, self.train_size)
        self.x_test, self.y_test, _ , _ = self.load_data(self.house_indicies_test, 1)
        logging.debug(f"Train -> input shape: {self.x_train.shape} | label shape: {self.y_train.shape}")
        logging.debug(f"Validation -> input shape: {self.x_val.shape} | label shape: {self.y_val.shape}")
        logging.debug(f"Test -> input shape: {self.x_test.shape} | label shape: {self.y_test.shape}")

    def batch_windowing(self, input_batch, output_batch):
        import sys
        I_ = []
        O_ = []
        nan_scartati = 0
        size_scartati = 0
        

        input_batch_time_index = input_batch.index
        # prepare input and output batch
        for start in range(0,len(input_batch)-self.window_len+1,self.window_len):
            end = start + self.window_len - 1
            input_window = input_batch[input_batch_time_index[start]:input_batch_time_index[end]].values
            output_window = output_batch[input_batch_time_index[start]:input_batch_time_index[end]].values

            # Controllo NaN
            if np.any(np.isnan(output_window)) or np.any(np.isnan(input_window)):
                nan_scartati += 1
                continue
            
            # Controllo dimensione
            if input_window.shape[0] != self.window_len or output_window.shape[0] != self.window_len:
                size_scartati += 1
                continue
            else:
                I_.append(input_window)
                O_.append(output_window)
        n_windows = (len(input_batch) - self.window_len) // self.window_len + 1
        # logging.debug(f"finestre totali: {n_windows} numero di finestre scartate per NaN: {nan_scartati} | numero di fiestre scartate per dimensione {size_scartati}")
        # GESTIONE LISTE VUOTE: se tutte le finestre sono state scartate
        if len(I_) == 0:
            # Restituisce array vuoti ma con la struttura corretta (0, window_size, 1)
            return np.empty((0, self.window_len, 1)), np.empty((0, self.window_len, 1))
        I_ = np.array(I_)
        O_ = np.array(O_)
        I_ = np.expand_dims(I_, axis=-1)
        O_ = np.expand_dims(O_, axis=-1)

        return I_, O_


    def load_data(self, house_indicies, train_split):
        I_train_full, O_train_full, I_val_full, O_val_full = [], [], [], []
        for house_id in house_indicies:
            house_data = pd.read_csv(os.path.join(self.data_location, f"casa_{house_id}_full.csv"))
            house_data[self.key] = pd.to_datetime(house_data[self.key], utc=True)
            house_data = house_data.set_index(self.key)
            house_data.index = house_data.index.tz_convert(self.fuso)

            # logging.debug(f"campioni della casa {house_id} prima del resample: {len(house_data)}")

            house_data = house_data.resample(str(self.resample_period)+'s').mean().ffill(limit=int(3600/self.resample_period))
            #effettuo il resample mediando i dati dentro i "period" secondi se ce ne sono di più, inoltre mantengo l'ultimo valore disponibile se ci sono buchi (ffill) minori di 1 ora di dati
            # logging.debug(f"campioni della casa {house_id} dopo il resample: {len(house_data)}")

            months = house_data.groupby([house_data.index.year, house_data.index.month])
            I_train, O_train, I_val, O_val = [], [], [], []
            for month, month_data in months:
                n = len(month_data)
                n_train = int(n*train_split)
                train_block = month_data.iloc[:n_train]   # primo 80% del mese
                val_block = month_data.iloc[n_train:]     # ultimo 20% del mese
                x, y = self.batch_windowing(train_block['P_ex'], train_block['P_pr'])
                I_train.append(x); O_train.append(y)

                x, y = self.batch_windowing(val_block['P_ex'], val_block['P_pr'])
                I_val.append(x); O_val.append(y)
                # logging.debug(f"\rMese: {month}")
            I_train_full.append(np.concatenate(I_train, axis=0))
            O_train_full.append(np.concatenate(O_train, axis=0))
            I_val_full.append(np.concatenate(I_val, axis=0))
            O_val_full.append(np.concatenate(O_val, axis=0))
        return np.concatenate(I_train_full), np.concatenate(O_train_full), np.concatenate(I_val_full), np.concatenate(O_val_full)

    def get_datasets(self):
        # Normalization of aggregate in [-1,1]
        self.x_min = self.x_train.min()
        self.x_max = self.x_train.max()
        x_train =  2 * (self.x_train-self.x_min)/(self.x_max-self.x_min) - 1
        x_val = 2 * (self.x_val-self.x_min)/(self.x_max-self.x_min) - 1
        x_test = 2 * (self.x_test-self.x_min)/(self.x_max-self.x_min) - 1

        # Normalization of appliances in [0,1]
        self.y_min = self.y_train.min()
        self.y_max = self.y_train.max()

        y_train = (self.y_train - self.y_min)/(self.y_max -self.y_min)
        y_val = (self.y_val - self.y_min) / (self.y_max - self.y_min)

        train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
        test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(self.y_test, dtype=torch.float32))

        return train_dataset, val_dataset, test_dataset, self.x_min, self.x_max, self.y_min, self.y_max

