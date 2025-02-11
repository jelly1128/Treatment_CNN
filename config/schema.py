from pydantic import BaseModel

class TrainingConfig(BaseModel):
    img_size: int
    num_classes: int
    model_architecture: str
    pretrained: bool
    freeze_backbone: bool
    learning_rate: float
    batch_size: int
    max_epochs: int

class TestConfig(BaseModel):
    img_size: int
    num_classes: int

class PathConfig(BaseModel):
    root: str
    save_dir: str
    model: str | None = None

class Config(BaseModel):
    training: TrainingConfig | None = None
    test: TestConfig | None = None
    paths: PathConfig