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

class TestConfig(BaseModel):
    mode: str
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