# NumPy 1.24+ compatibility fix for detectron2
# Restore deprecated aliases removed in NumPy 1.24
import warnings
import numpy as np
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    _np_bool_exists = hasattr(np, 'bool') and isinstance(getattr(np, 'bool', None), type)
if not _np_bool_exists:
    np.bool = np.bool_
    np.int = np.int_
    np.float = np.float_
    np.complex = np.complex_
    np.object = np.object_
    np.str = np.str_

import os
from detectron2.utils import comm
from detectron2.engine import launch
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from lfsd.config import get_cfg, set_global_cfg
from lfsd.evaluation import DatasetEvaluators, verify_results
from lfsd.engine import Trainer, TwoSteamTrainer, default_argument_parser, default_setup


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    if cfg.DATASETS.TWO_STREAM:
        trainer = TwoSteamTrainer(cfg)
    else:
        trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
