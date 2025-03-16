import h5py
import numpy as np
import os

def load_from_hdf5(file_path):
    """
    Загружает данные из HDF5 файла
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден")
        
    print(f"Загрузка данных из {file_path}...")
    with h5py.File(file_path, 'r') as hf:
        frames = np.array(hf['frames'])
        actions = np.array(hf['actions'])
    
    print(f"Загружено {len(frames)} кадров и {len(actions)} действий")
    return frames, actions 