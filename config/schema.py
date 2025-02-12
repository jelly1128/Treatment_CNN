from pydantic import BaseModel, RootModel

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
    dataset_root: str
    save_dir: str
    model: str | None = None
    
class SplitConfig(RootModel[dict[str, list[str]]]):
    pass

class Config(BaseModel):
    training: TrainingConfig | None = None
    test: TestConfig | None = None
    paths: PathConfig
    splits: SplitConfig | None = None