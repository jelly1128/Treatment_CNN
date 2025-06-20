import torchvision.transforms as transforms

def get_train_transforms():
    return transforms.Compose([
        # transforms.RandomAffine(degrees=5, translate=(0.025, 0.025), shear=5),
        transforms.RandomRotation(degrees=360),
        transforms.ToTensor(),
    ])

def get_test_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
    ])