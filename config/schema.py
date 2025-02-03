from pydantic import BaseModel

class TrainingConfig(BaseModel):
    mode: str
    img_size: int
    num_classes: int
    pretrained: bool
    freeze_backbone: bool
    learning_rate: float
    batch_size: int
    max_epochs: int

# class TestConfig(BaseModel):
#     mode: str
#     img_size: int
#     num_classes: int
    

class PathConfig(BaseModel):
    root: str
    save_dir: str

class Config(BaseModel):
    training: TrainingConfig
    paths: PathConfig

# class PathConfig(BaseModel):
#     root: str
#     model: str
#     save_name: str

# class Config(BaseModel):
#     test: TestConfig
#     paths: PathConfig