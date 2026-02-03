import os
import pandas as pd
import datasets


class MM4Config(datasets.BuilderConfig):
    """BuilderConfig for MM4."""

    def __init__(self, **kwargs):
        super(MM4Config, self).__init__(**kwargs)


class MM4(datasets.GeneratorBasedBuilder):
    """MM4 dataset."""

    BUILDER_CONFIGS = [
        MM4Config(
            name="mm4",
            version=datasets.Version("1.0.0"),
            description="MM4 multimodal benchmark dataset",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="MM4 dataset",
            features=datasets.Features(
                {
                    "index": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "URL": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                    "figure_name": datasets.Value("string"),
                    "category": datasets.Value("string"),
                    "image": datasets.Image(),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "csv_path": "/lpai/volumes/so-volume-bd-ga/lhp/datasets/MM4/MM4.csv",
                    "images_dir": "/lpai/volumes/so-volume-bd-ga/lhp/datasets/MM4/images",
                },
            ),
        ]

    def _generate_examples(self, csv_path, images_dir):
        """Yields examples."""
        df = pd.read_csv(csv_path)

        for idx, row in df.iterrows():
            image_path = os.path.join(images_dir, row["figure_name"])

            yield idx, {
                "index": str(row["index"]),
                "question": str(row["question"]),
                "URL": str(row.get("URL", "")),
                "answer": str(row["answer"]),
                "figure_name": str(row["figure_name"]),
                "category": str(row["category"]),
                "image": image_path,
            }
