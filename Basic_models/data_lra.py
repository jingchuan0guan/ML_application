import os
import numpy as np
from PIL import Image
# /home/kan/ML_application/s4/data/pathfinder/pathfinder32/curv_baseline/metadata/
def load_pathfider_metafiles(
    img_size = [32, 64, 128, 256][0],
    difficulty = ["E", "M", "H"][0], # easy, intermediate, hard
    ROOT_DIR = "", ##### change here
    ):
    DIFFICULT_DIR={"E":"curv_baseline/", "M":"curv_contour_length_9/", "H":"curv_contour_length_14/"}
    BASE_DIR = ROOT_DIR + f"pathfinder{img_size}/" + DIFFICULT_DIR[difficulty]
    METADATA_DIR = os.path.join(BASE_DIR, 'metadata')
    meta_files = [os.path.join(METADATA_DIR, fname) for fname in os.listdir(METADATA_DIR) if fname.endswith('.npy')]

    image_paths = []
    labels = []
    _load_counter = 0
    for fname in os.listdir(METADATA_DIR):
        if not fname.endswith('.npy'):
            continue
        with open(os.path.join(METADATA_DIR, fname), 'r') as f:
            lines = f.readlines()
        len_metafile = len(lines)
        # print("row len", len_metafile)
        
        for line in lines:
            parts = line.strip().split(' ')
            # print("row", parts)
            # if len(parts) < 4:
            #     continue  # skip malformed lines
            
            dir_name, file_name, label = parts[0], parts[1], int(parts[3])
            image_path = os.path.join(BASE_DIR, dir_name, file_name)
            image_paths.append(image_path)
            labels.append(label)
        
        _load_counter += len_metafile
    
    print("metadata load counter:", _load_counter, " This is the number of all available files")
    return np.array(image_paths), np.array(labels)
# images = [np.array(Image.open(p)) for p in image_paths]
# print(f"Loaded {len(images)} images. First shape: {images[0].shape}, First label: {labels[0]}")

class PathFinderDataLoader:
    def __init__(
        self, num_samples, image_paths, labels, batch_size=32, seed=0, shuffle=True, normalize=True, drop_last=True,):
        self.image_paths=image_paths
        self.labels=labels
        """
        batch_size: int
        shuffle: bool
        drop_last: bool - if True, drop final batch if it's smaller than batch_size
        """
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalize = normalize
        self.drop_last = drop_last
        
        self.num_samples = num_samples
        self.indices = np.arange(self.num_samples)
        self.current_idx = 0

        if self.shuffle:
            np.random.seed(seed)
            np.random.shuffle(self.indices)
            # print(self.indices)

    def __iter__(self):
        return self.current_idx

    def __next__(self):
        if self.current_idx >= self.num_samples:
            raise StopIteration
        end_idx = self.current_idx + self.batch_size
        if end_idx > self.num_samples:
            if self.drop_last:
                raise StopIteration
            end_idx = self.num_samples

        batch_indices = self.indices[self.current_idx:end_idx] #np.array(self.indices[self.current_idx:end_idx], dtype=np.int16)
        # print("batch_indices", batch_indices)
        batch_labels = self.labels[batch_indices]
        batch_data = list(
            map( lambda i: np.array(Image.open(self.image_paths[i])), batch_indices )
            )
        batch_labels, batch_data = np.array(batch_labels).reshape(-1, 1), np.array(batch_data)
        # print("max value", np.max(batch_data))
        if self.normalize:
            batch_data = batch_data.astype(np.float32) / 255.0
        self.current_idx = end_idx
        return batch_data, batch_labels

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size

