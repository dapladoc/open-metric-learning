import hydra
from omegaconf import DictConfig

from oml.const import HYDRA_VERSION
from oml.lightning.pipelines.train import extractor_training_pipeline


@hydra.main(config_path=".", config_name="train_sop.yaml", version_base=HYDRA_VERSION)
def main_hydra(cfg: DictConfig) -> None:
    print("Training model on SOP dataset.")
    extractor_training_pipeline(cfg)


if __name__ == "__main__":
    main_hydra()
