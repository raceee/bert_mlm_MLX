class MLXDataset:
    def __init__(self, data, target, feature_names, target_names):
        self.data = data
        self.target = target
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]