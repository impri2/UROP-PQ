from beir import util
from beir.datasets.data_loader import GenericDataLoader

import pathlib, os
def load_beir_dataset(dataset,split):
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    return {'corpus':corpus,'queries':queries,'qrels':qrels}
if __name__ == '__main__':
    load_beir_dataset('scifact','train')