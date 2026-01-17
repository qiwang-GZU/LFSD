import glob
import os

from .meta_voc import register_meta_voc
from .meta_coco import register_meta_coco
from .meta_bcgd import register_meta_bcgd
from .builtin_meta import _get_builtin_metadata
from detectron2.data import DatasetCatalog, MetadataCatalog


def _get_datasets_root():
    """Get dataset root directory, prioritizing DETECTRON2_DATASETS environment variable."""

    env_root = os.environ.get("DETECTRON2_DATASETS")
    if env_root:
        return env_root

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidate_roots = [
        os.path.join(repo_root, "datasets"),
        os.path.join(repo_root, "..", "LFSD-EAEC", "datasets"),
        "/nfs/home/fragenmgr/code/LFSD-EAEC/datasets",
        "datasets",
    ]

    for root in candidate_roots:
        abs_root = os.path.abspath(root)
        if os.path.isdir(abs_root):
            return abs_root

    return "datasets"


# -------- COCO -------- #
def register_all_coco(root=None):
    if root is None:
        root = _get_datasets_root()

    METASPLITS = [
        ("coco14_trainval_all", "coco/trainval2014", "cocosplit/datasplit/trainvalno5k.json"),
        ("coco14_trainval_base", "coco/trainval2014", "cocosplit/datasplit/trainvalno5k.json"),
        ("coco14_test_all", "coco/val2014", "cocosplit/datasplit/5k.json"),
        ("coco14_test_base", "coco/val2014", "cocosplit/datasplit/5k.json"),
        ("coco14_test_novel", "coco/val2014", "cocosplit/datasplit/5k.json"),
        ("removecoco14_trainval_all", "coco/trainval2014", "cocosplit/datasplit/trainvalno5k.json"),
    ]
    for prefix in ["all", "novel"]:
        for shot in [1, 2, 3, 5, 10, 30]:
            for seed in range(10):
                name = "coco14_trainval_{}_{}shot_seed{}".format(prefix, shot, seed)
                METASPLITS.append((name, "coco/trainval2014", ""))

                if prefix == "all":
                    name = "removecoco14_trainval_{}_{}shot_seed{}".format(prefix, shot, seed)
                    METASPLITS.append((name, "coco/trainval2014", ""))

    for name, imgdir, annofile in METASPLITS:
        register_meta_coco(
            name,
            _get_builtin_metadata("coco_fewshot"),
            os.path.join(root, imgdir),
            os.path.join(root, annofile),
        )


# -------- PASCAL VOC -------- #
def register_all_voc(root=None):
    if root is None:
        root = _get_datasets_root()

    METASPLITS = [
        ("voc_2007_trainval_base1", "VOC2007", "trainval", "base1", 1),
        ("voc_2007_trainval_base2", "VOC2007", "trainval", "base2", 2),
        ("voc_2007_trainval_base3", "VOC2007", "trainval", "base3", 3),
        ("voc_2012_trainval_base1", "VOC2012", "trainval", "base1", 1),
        ("voc_2012_trainval_base2", "VOC2012", "trainval", "base2", 2),
        ("voc_2012_trainval_base3", "VOC2012", "trainval", "base3", 3),
        ("voc_2007_trainval_all1", "VOC2007", "trainval", "base_novel_1", 1),
        ("voc_2007_trainval_all2", "VOC2007", "trainval", "base_novel_2", 2),
        ("voc_2007_trainval_all3", "VOC2007", "trainval", "base_novel_3", 3),
        ("voc_2012_trainval_all1", "VOC2012", "trainval", "base_novel_1", 1),
        ("voc_2012_trainval_all2", "VOC2012", "trainval", "base_novel_2", 2),
        ("voc_2012_trainval_all3", "VOC2012", "trainval", "base_novel_3", 3),
        ("voc_2007_test_base1", "VOC2007", "test", "base1", 1),
        ("voc_2007_test_base2", "VOC2007", "test", "base2", 2),
        ("voc_2007_test_base3", "VOC2007", "test", "base3", 3),
        ("voc_2007_test_novel1", "VOC2007", "test", "novel1", 1),
        ("voc_2007_test_novel2", "VOC2007", "test", "novel2", 2),
        ("voc_2007_test_novel3", "VOC2007", "test", "novel3", 3),
        ("voc_2007_test_all1", "VOC2007", "test", "base_novel_1", 1),
        ("voc_2007_test_all2", "VOC2007", "test", "base_novel_2", 2),
        ("voc_2007_test_all3", "VOC2007", "test", "base_novel_3", 3),
    ]
    for prefix in ["all", "novel"]:
        for sid in range(1, 4):
            for shot in [1, 2, 3, 5, 10]:
                for year in [2007, 2012]:
                    for seed in range(30):
                        seed = "_seed{}".format(seed)
                        name = "voc_{}_trainval_{}{}_{}shot{}".format(
                            year, prefix, sid, shot, seed
                        )
                        dirname = "VOC{}".format(year)
                        img_file = "{}_{}shot_split_{}_trainval".format(
                            prefix, shot, sid
                        )
                        keepclasses = (
                            "base_novel_{}".format(sid)
                            if prefix == "all"
                            else "novel{}".format(sid)
                        )
                        METASPLITS.append(
                            (name, dirname, img_file, keepclasses, sid)
                        )

                        if prefix == "all":
                            name = "removevoc_{}_trainval_{}{}_{}shot{}".format(
                                year, prefix, sid, shot, seed
                            )
                            METASPLITS.append(
                                (name, dirname, img_file, keepclasses, sid)
                            )

    for name, dirname, split, keepclasses, sid in METASPLITS:
        year = 2007 if "2007" in name else 2012
        register_meta_voc(
            name,
            _get_builtin_metadata("voc_fewshot"),
            os.path.join(root, dirname),
            split,
            year,
            keepclasses,
            sid,
        )
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"

def register_all_bcgd(root=None):
    if root is None:
        root = _get_datasets_root()

    dataset_root = os.path.join(root, "BloodCell-Detection-Datatset-main")
    ann_root = os.path.join(root, "bcgd", "annotations")
    metadata = _get_builtin_metadata("bcgd_fewshot")

    splits = [
        ("bcgd_train_full", os.path.join(dataset_root, "train", "images"), os.path.join(ann_root, "bcgd_train_full.json")),
        ("bcgd_valid_full", os.path.join(dataset_root, "valid", "images"), os.path.join(ann_root, "bcgd_valid_full.json")),
        ("bcgd_test_full", os.path.join(dataset_root, "test", "images"), os.path.join(ann_root, "bcgd_test_full.json")),
    ]

    for name, imgdir, annofile in splits:
        register_meta_bcgd(name, metadata, imgdir, annofile)

    few_shot_pattern = os.path.join(ann_root, "bcgd_train_*shot_seed*.json")
    for json_path in sorted(glob.glob(few_shot_pattern)):
        dataset_name = os.path.splitext(os.path.basename(json_path))[0]
        imgdir = os.path.join(dataset_root, "train", "images")
        register_meta_bcgd(dataset_name, metadata, imgdir, json_path)


register_all_coco()
register_all_voc()
register_all_bcgd()
