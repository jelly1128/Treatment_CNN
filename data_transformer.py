import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class DataTransformer:
    @staticmethod
    def get_olympus_transforms(img_size):
        return {
            'train_val': transforms.Compose([
                transforms.Lambda(lambda x: TF.crop(x, top=20, left=710, height=1040, width=1180)),
                transforms.Resize((img_size, img_size)),
                transforms.RandomRotation(degrees=360),
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                transforms.Lambda(lambda x: TF.crop(x, top=20, left=710, height=1040, width=1180)),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])
        }

    @staticmethod
    def get_fujifilm_transforms(img_size):
        return {
            'train_val': transforms.Compose([
                transforms.Lambda(lambda x: TF.crop(x, top=25, left=330, height=970, width=1260)),
                transforms.Resize((img_size, img_size)),
                transforms.RandomRotation(degrees=360),
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                transforms.Lambda(lambda x: TF.crop(x, top=25, left=330, height=970, width=1260)),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])
        }
        
    @staticmethod
    def get_transform(maker: str, phase: str, img_size: int):
        if maker == 'olympus':
            if phase == 'train':
                return transforms.Compose([
                        transforms.Lambda(lambda x: TF.crop(x, top=20, left=710, height=1040, width=1180)),
                        transforms.Resize((img_size, img_size)),
                        transforms.RandomRotation(degrees=360),
                        transforms.ToTensor()
                ])
                
                
            elif phase == 'val' or phase == 'test':
                return transforms.Compose([
                        transforms.Lambda(lambda x: TF.crop(x, top=20, left=710, height=1040, width=1180)),
                        transforms.Resize((img_size, img_size)),
                        transforms.ToTensor()
                    ])
        elif maker == 'fujifilm':
            if phase == 'train':
                return transforms.Compose([
                        transforms.Lambda(lambda x: TF.crop(x, top=25, left=330, height=970, width=1260)),
                        transforms.Resize((img_size, img_size)),
                        transforms.RandomRotation(degrees=360),
                        transforms.ToTensor()
                    ])
                
            elif phase == 'val' or phase == 'test':
                return transforms.Compose([
                        transforms.Lambda(lambda x: TF.crop(x, top=25, left=330, height=970, width=1260)),
                        transforms.Resize((img_size, img_size)),
                        transforms.ToTensor()
                    ])