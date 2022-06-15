class SiameseDataset(Dataset):
    def __init__(self, df, img_size = (224,224), normalize_img = False, in_channel = 3, is_valid=False, transform=None):
        self.files = list(df['file_path'])
        self.labels = list(df['pair_num'])
        # Create a dictionary mapping pair numbers to their files
        self.lbl2files = {l: [self.files[i] for i in range(len(df)) if self.labels[i] == l] for l in self.labels}
        self.is_valid = is_valid
        self.transform = transform
        self.normalize_img = normalize_img
        self.im_mode = "L" if (in_channel == 1) else "RGB"
        self.basic_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Resize(img_size),
        ])

        if is_valid: self.files2 = [self._draw(f) for f in self.files]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        file1 = self.files[i]
        (file2, same) = self.files2[i] if self.is_valid else self._draw(file1)
        if (same == 0):
            same = -1
        img1,img2 = self.get_image(file1),self.get_image(file2)
        return ((img1, img2), torch.Tensor([same]).squeeze())
    
    def _draw(self, f):
        same = random.random() < 0.5
        cls = self.labels[self.files.index(f)]
        if not same: cls = random.choice([l for l in self.labels if l != cls]) 
        return random.choice(self.lbl2files[cls]),same
    
    def get_image(self, image_path):
        image = Image.open(image_path).convert(self.im_mode)
        image = self.basic_transforms(image)
        if self.transform:
            image = self.transform(image)
        if self.normalize_img:
            image = image/255.
        return image