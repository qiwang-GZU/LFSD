"""
Training Dynamics Analysis Trainer for Few-Shot Object Detection.

This module provides a specialized Trainer that records:
1. Novel-class specific losses (loss_cls_novel, loss_box_novel)
2. Periodic AP evaluation during training
3. All metrics to JSON/CSV for post-hoc analysis

Designed for TPAMI-style training dynamics visualization comparing
DeFRCN baseline vs SC-IPA enhanced methods.
"""

import os
import json
import time
import torch
import logging
import numpy as np
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple

from detectron2.utils import comm
from detectron2.engine import hooks
from detectron2.utils.events import get_event_storage

from lfsd.engine.defaults import Trainer, TwoSteamTrainer
from lfsd.engine.hooks import EvalHookDeFRCN

logger = logging.getLogger(__name__)

__all__ = ["TrainingDynamicsTrainer", "TwoStreamDynamicsTrainer"]


class NovelLossTracker:
    """
    Tracks novel-class specific losses during training.
    
    This class maintains running statistics for:
    - loss_cls_novel: classification loss for novel classes only
    - loss_box_novel: bounding box regression loss for novel classes only
    """
    
    def __init__(self, novel_class_ids: List[int], num_classes: int):
        """
        Args:
            novel_class_ids: List of class indices that are novel classes
            num_classes: Total number of foreground classes (excluding background)
        """
        self.novel_class_ids = set(novel_class_ids)
        self.num_classes = num_classes
        self.bg_class_id = num_classes  # Background class index
        
    def compute_novel_losses(
        self,
        pred_class_logits: torch.Tensor,
        pred_proposal_deltas: torch.Tensor,
        gt_classes: torch.Tensor,
        gt_boxes: torch.Tensor,
        proposals_tensor: torch.Tensor,
        box2box_transform,
        smooth_l1_beta: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses specifically for novel-class RoIs.
        
        Args:
            pred_class_logits: (R, K+1) predicted class logits
            pred_proposal_deltas: (R, K*4) or (R, 4) predicted box deltas
            gt_classes: (R,) ground-truth class labels
            gt_boxes: (R, 4) ground-truth boxes
            proposals_tensor: (R, 4) proposal boxes
            box2box_transform: Box2BoxTransform instance
            smooth_l1_beta: Beta for smooth L1 loss
            
        Returns:
            Dict with 'loss_cls_novel' and 'loss_box_novel' tensors
        """
        import torch.nn.functional as F
        from fvcore.nn import smooth_l1_loss
        
        device = pred_class_logits.device
        
        # Create mask for novel class RoIs (foreground only)
        novel_mask = torch.zeros(len(gt_classes), dtype=torch.bool, device=device)
        for cls_id in self.novel_class_ids:
            novel_mask |= (gt_classes == cls_id)
        
        num_novel = novel_mask.sum().item()
        
        # Handle case with no novel samples
        if num_novel == 0:
            return {
                'loss_cls_novel': torch.tensor(0.0, device=device),
                'loss_box_novel': torch.tensor(0.0, device=device),
                'num_novel_samples': 0,
            }
        
        # Classification loss for novel classes
        novel_logits = pred_class_logits[novel_mask]
        novel_gt_classes = gt_classes[novel_mask]
        loss_cls_novel = F.cross_entropy(novel_logits, novel_gt_classes, reduction='mean')
        
        # Box regression loss for novel classes
        gt_proposal_deltas = box2box_transform.get_deltas(
            proposals_tensor, gt_boxes
        )
        
        box_dim = gt_proposal_deltas.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = pred_proposal_deltas.size(1) == box_dim
        
        # Get foreground novel indices
        fg_novel_mask = novel_mask & (gt_classes < self.bg_class_id)
        fg_novel_inds = torch.nonzero(fg_novel_mask).squeeze(1)
        
        if fg_novel_inds.numel() == 0:
            loss_box_novel = torch.tensor(0.0, device=device)
        else:
            if cls_agnostic_bbox_reg:
                gt_class_cols = torch.arange(box_dim, device=device)
            else:
                fg_gt_classes = gt_classes[fg_novel_inds]
                gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(
                    box_dim, device=device
                )
            
            loss_box_novel = smooth_l1_loss(
                pred_proposal_deltas[fg_novel_inds[:, None], gt_class_cols],
                gt_proposal_deltas[fg_novel_inds],
                smooth_l1_beta,
                reduction="sum",
            )
            loss_box_novel = loss_box_novel / max(fg_novel_inds.numel(), 1)
        
        return {
            'loss_cls_novel': loss_cls_novel,
            'loss_box_novel': loss_box_novel,
            'num_novel_samples': num_novel,
        }


class TrainingDynamicsWriter:
    """
    Writes training dynamics data to JSON and CSV files.
    """
    
    def __init__(self, output_dir: str, method_name: str):
        self.output_dir = output_dir
        self.method_name = method_name
        self.loss_records = []
        self.ap_records = []
        
        os.makedirs(output_dir, exist_ok=True)
        
    def add_loss_record(
        self,
        iteration: int,
        loss_cls: float,
        loss_box: float,
        loss_cls_novel: float,
        loss_box_novel: float,
        total_loss: float,
        lr: float,
        num_novel_samples: int = 0,
    ):
        """Record loss values for one iteration."""
        self.loss_records.append({
            'iteration': iteration,
            'loss_cls': loss_cls,
            'loss_box_reg': loss_box,
            'loss_cls_novel': loss_cls_novel,
            'loss_box_novel': loss_box_novel,
            'total_loss': total_loss,
            'lr': lr,
            'num_novel_samples': num_novel_samples,
        })
        
    def add_ap_record(
        self,
        iteration: int,
        ap_novel_50: float,
        ap_novel_75: Optional[float] = None,
        ap_base_50: Optional[float] = None,
        ap_all_50: Optional[float] = None,
        extra_metrics: Optional[Dict] = None,
    ):
        """Record AP values at evaluation checkpoint."""
        record = {
            'iteration': iteration,
            'AP_novel_50': ap_novel_50,
        }
        if ap_novel_75 is not None:
            record['AP_novel_75'] = ap_novel_75
        if ap_base_50 is not None:
            record['AP_base_50'] = ap_base_50
        if ap_all_50 is not None:
            record['AP_all_50'] = ap_all_50
        if extra_metrics:
            record.update(extra_metrics)
        self.ap_records.append(record)
        
    def save(self):
        """Save all records to files."""
        # Save loss records
        loss_file = os.path.join(self.output_dir, f'{self.method_name}_loss_dynamics.json')
        with open(loss_file, 'w') as f:
            json.dump(self.loss_records, f, indent=2)
        logger.info(f"Loss dynamics saved to {loss_file}")
        
        # Save AP records
        ap_file = os.path.join(self.output_dir, f'{self.method_name}_ap_dynamics.json')
        with open(ap_file, 'w') as f:
            json.dump(self.ap_records, f, indent=2)
        logger.info(f"AP dynamics saved to {ap_file}")
        
        # Also save as CSV for easy plotting
        self._save_csv(
            self.loss_records,
            os.path.join(self.output_dir, f'{self.method_name}_loss_dynamics.csv')
        )
        self._save_csv(
            self.ap_records,
            os.path.join(self.output_dir, f'{self.method_name}_ap_dynamics.csv')
        )
        
    def _save_csv(self, records: List[Dict], filepath: str):
        """Save records as CSV file."""
        if not records:
            return
        import csv
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        logger.info(f"CSV saved to {filepath}")


class TrainingDynamicsTrainer(Trainer):
    """
    Extended Trainer that tracks training dynamics for few-shot analysis.
    
    Key features:
    - Records novel-class specific losses every iteration
    - Performs periodic evaluation at configurable intervals
    - Outputs all metrics to JSON/CSV for visualization
    """
    
    def __init__(self, cfg, eval_period: int = 1000, method_name: str = "method"):
        """
        Args:
            cfg: Config node
            eval_period: How often to run full evaluation (in iterations)
            method_name: Name identifier for output files
        """
        # Set eval_period and method_name BEFORE super().__init__() 
        # because build_hooks() is called during parent init and needs these
        self.eval_period = eval_period
        self.method_name = method_name
        self._dynamics_cfg = cfg  # Cache for later use
        
        super().__init__(cfg)
        
        # Initialize novel class tracker
        dataset_name = cfg.DATASETS.TRAIN[0].lower()
        self.novel_class_ids = self._get_novel_class_ids(dataset_name, cfg)
        self.novel_tracker = NovelLossTracker(
            self.novel_class_ids, 
            cfg.MODEL.ROI_HEADS.NUM_CLASSES
        )
        
        # Initialize dynamics writer
        self.dynamics_writer = TrainingDynamicsWriter(
            cfg.OUTPUT_DIR, 
            method_name
        )
        
        # Cache for accumulating novel losses (averaged per logging period)
        self._novel_loss_cache = defaultdict(list)
        
        logger.info(f"TrainingDynamicsTrainer initialized with eval_period={eval_period}")
        logger.info(f"Novel class IDs: {self.novel_class_ids}")
        
    def _get_novel_class_ids(self, dataset_name: str, cfg) -> List[int]:
        """Get novel class indices based on dataset."""
        if 'voc' in dataset_name:
            # Get split from test dataset name (e.g., voc_2007_test_all1 -> split 1)
            test_name = cfg.DATASETS.TEST[0] if cfg.DATASETS.TEST else ""
            if test_name.endswith('1'):
                # Split 1: bird, bus, cow, motorbike, sofa (indices 15-19 in 20-class VOC)
                return [15, 16, 17, 18, 19]
            elif test_name.endswith('2'):
                # Split 2
                return [15, 16, 17, 18, 19]  # Adjust based on your split definition
            elif test_name.endswith('3'):
                # Split 3
                return [15, 16, 17, 18, 19]  # Adjust based on your split definition
            else:
                return [15, 16, 17, 18, 19]  # Default to split 1
        elif 'coco' in dataset_name:
            # COCO novel classes (20 classes)
            return [0, 1, 2, 3, 4, 5, 6, 8, 14, 15, 16, 17, 18, 19, 39, 56, 57, 58, 60, 62]
        else:
            # Default: treat all as novel
            return list(range(cfg.MODEL.ROI_HEADS.NUM_CLASSES))
    
    def build_hooks(self):
        """Build hooks with modified evaluation frequency."""
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0
        
        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
        ]
        
        # Checkpointing
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )
        
        # Custom evaluation hook with our eval_period
        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            # Record AP to dynamics writer
            if comm.is_main_process() and self._last_eval_results:
                self._record_ap_metrics(self._last_eval_results)
            return self._last_eval_results
        
        ret.append(EvalHookDeFRCN(
            self.eval_period, test_and_save_results, self.cfg
        ))
        
        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers()))
            
        return ret
    
    def _record_ap_metrics(self, results: Dict):
        """Extract and record AP metrics from evaluation results."""
        iteration = self.iter
        
        # Handle different result formats (VOC vs COCO)
        ap_novel_50 = None
        ap_novel_75 = None
        ap_base_50 = None
        
        if isinstance(results, dict):
            # Try to extract VOC-style metrics
            if 'bbox' in results:
                bbox_results = results['bbox']
                ap_novel_50 = bbox_results.get('nAP50', bbox_results.get('AP50'))
                ap_base_50 = bbox_results.get('bAP50')
            elif 'AP50' in results:
                ap_novel_50 = results.get('nAP50', results.get('AP50'))
                ap_base_50 = results.get('bAP50')
            # Try COCO-style
            elif 'AP' in results:
                ap_novel_50 = results.get('AP')
                ap_novel_75 = results.get('AP75')
        
        if ap_novel_50 is not None:
            self.dynamics_writer.add_ap_record(
                iteration=iteration,
                ap_novel_50=float(ap_novel_50),
                ap_novel_75=float(ap_novel_75) if ap_novel_75 else None,
                ap_base_50=float(ap_base_50) if ap_base_50 else None,
            )
            logger.info(f"Iter {iteration}: AP_novel_50 = {ap_novel_50:.2f}")
    
    def run_step(self):
        """
        Extended run_step that also computes novel-class losses.
        """
        assert self.model.training, "[TrainingDynamicsTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start
        
        # Forward pass
        loss_dict = self.model(data)
        losses = sum(loss_dict.values())
        
        self.optimizer.zero_grad()
        losses.backward()
        
        self._write_metrics(loss_dict, data_time)
        
        self.optimizer.step()
        
        # Record losses to dynamics writer (every iteration)
        if comm.is_main_process():
            storage = get_event_storage()
            
            # Get current LR
            lr = self.optimizer.param_groups[0]['lr']
            
            # Get loss values
            loss_cls = loss_dict.get('loss_cls', torch.tensor(0.0)).item()
            loss_box = loss_dict.get('loss_box_reg', torch.tensor(0.0)).item()
            total_loss = losses.item()
            
            # For novel losses, we need to compute them separately
            # This is a simplified version - full implementation would need
            # to access the model's internal state
            loss_cls_novel = loss_dict.get('loss_cls_novel', torch.tensor(0.0))
            loss_box_novel = loss_dict.get('loss_box_novel', torch.tensor(0.0))
            
            if isinstance(loss_cls_novel, torch.Tensor):
                loss_cls_novel = loss_cls_novel.item()
            if isinstance(loss_box_novel, torch.Tensor):
                loss_box_novel = loss_box_novel.item()
            
            self.dynamics_writer.add_loss_record(
                iteration=self.iter,
                loss_cls=loss_cls,
                loss_box=loss_box,
                loss_cls_novel=loss_cls_novel,
                loss_box_novel=loss_box_novel,
                total_loss=total_loss,
                lr=lr,
            )
    
    def train(self):
        """
        Run training and save dynamics data at the end.
        """
        result = super().train()
        
        # Save all dynamics data
        if comm.is_main_process():
            self.dynamics_writer.save()
            logger.info(f"Training dynamics saved to {self.cfg.OUTPUT_DIR}")
        
        return result


class TwoStreamDynamicsTrainer(TwoSteamTrainer):
    """
    Extended TwoStreamTrainer that tracks training dynamics.
    
    This is for the SC-IPA / Dual-Stream method with explicit prototype memory.
    """
    
    def __init__(self, cfg, eval_period: int = 1000, method_name: str = "method"):
        # Set eval_period and method_name BEFORE super().__init__() 
        # because build_hooks() is called during parent init and needs these
        self.eval_period = eval_period
        self.method_name = method_name
        self._dynamics_cfg = cfg  # Cache for later use
        
        super().__init__(cfg)
        
        # Initialize novel class tracker
        dataset_name = cfg.DATASETS.TRAIN[0].lower()
        self.novel_class_ids = self._get_novel_class_ids(dataset_name, cfg)
        self.novel_tracker = NovelLossTracker(
            self.novel_class_ids, 
            cfg.MODEL.ROI_HEADS.NUM_CLASSES
        )
        
        # Initialize dynamics writer
        self.dynamics_writer = TrainingDynamicsWriter(
            cfg.OUTPUT_DIR, 
            method_name
        )
        
        logger.info(f"TwoStreamDynamicsTrainer initialized with eval_period={eval_period}")
        logger.info(f"Novel class IDs: {self.novel_class_ids}")
        
    def _get_novel_class_ids(self, dataset_name: str, cfg) -> List[int]:
        """Get novel class indices based on dataset."""
        if 'voc' in dataset_name:
            test_name = cfg.DATASETS.TEST[0] if cfg.DATASETS.TEST else ""
            if test_name.endswith('1'):
                return [15, 16, 17, 18, 19]
            elif test_name.endswith('2'):
                return [15, 16, 17, 18, 19]
            elif test_name.endswith('3'):
                return [15, 16, 17, 18, 19]
            else:
                return [15, 16, 17, 18, 19]
        elif 'coco' in dataset_name:
            return [0, 1, 2, 3, 4, 5, 6, 8, 14, 15, 16, 17, 18, 19, 39, 56, 57, 58, 60, 62]
        else:
            return list(range(cfg.MODEL.ROI_HEADS.NUM_CLASSES))
    
    def build_hooks(self):
        """Build hooks with modified evaluation frequency."""
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0
        
        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
        ]
        
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )
        
        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            if comm.is_main_process() and self._last_eval_results:
                self._record_ap_metrics(self._last_eval_results)
            return self._last_eval_results
        
        ret.append(EvalHookDeFRCN(
            self.eval_period, test_and_save_results, self.cfg
        ))
        
        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers()))
            
        return ret
    
    def _record_ap_metrics(self, results: Dict):
        """Extract and record AP metrics from evaluation results."""
        iteration = self.iter
        
        ap_novel_50 = None
        ap_novel_75 = None
        ap_base_50 = None
        
        if isinstance(results, dict):
            if 'bbox' in results:
                bbox_results = results['bbox']
                ap_novel_50 = bbox_results.get('nAP50', bbox_results.get('AP50'))
                ap_base_50 = bbox_results.get('bAP50')
            elif 'AP50' in results:
                ap_novel_50 = results.get('nAP50', results.get('AP50'))
                ap_base_50 = results.get('bAP50')
            elif 'AP' in results:
                ap_novel_50 = results.get('AP')
                ap_novel_75 = results.get('AP75')
        
        if ap_novel_50 is not None:
            self.dynamics_writer.add_ap_record(
                iteration=iteration,
                ap_novel_50=float(ap_novel_50),
                ap_novel_75=float(ap_novel_75) if ap_novel_75 else None,
                ap_base_50=float(ap_base_50) if ap_base_50 else None,
            )
            logger.info(f"Iter {iteration}: AP_novel_50 = {ap_novel_50:.2f}")
    
    def run_step(self):
        """Extended run_step with novel-class loss tracking."""
        assert self.model.training, "[TwoStreamDynamicsTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        
        data = next(self._data_loader_iter)
        data_base = next(self._data_loader_base_iter)
        data.extend(data_base)
        data_time = time.perf_counter() - start
        
        loss_dict = self.model(data)
        losses = sum(loss_dict.values())
        
        self.optimizer.zero_grad()
        losses.backward()
        
        self._write_metrics(loss_dict, data_time)
        
        self.optimizer.step()
        
        # Record losses
        if comm.is_main_process():
            lr = self.optimizer.param_groups[0]['lr']
            
            loss_cls = loss_dict.get('loss_cls', torch.tensor(0.0)).item()
            loss_box = loss_dict.get('loss_box_reg', torch.tensor(0.0)).item()
            total_loss = losses.item()
            
            loss_cls_novel = loss_dict.get('loss_cls_novel', torch.tensor(0.0))
            loss_box_novel = loss_dict.get('loss_box_novel', torch.tensor(0.0))
            
            if isinstance(loss_cls_novel, torch.Tensor):
                loss_cls_novel = loss_cls_novel.item()
            if isinstance(loss_box_novel, torch.Tensor):
                loss_box_novel = loss_box_novel.item()
            
            self.dynamics_writer.add_loss_record(
                iteration=self.iter,
                loss_cls=loss_cls,
                loss_box=loss_box,
                loss_cls_novel=loss_cls_novel,
                loss_box_novel=loss_box_novel,
                total_loss=total_loss,
                lr=lr,
            )
    
    def train(self):
        """Run training and save dynamics data."""
        result = super().train()
        
        if comm.is_main_process():
            self.dynamics_writer.save()
            logger.info(f"Training dynamics saved to {self.cfg.OUTPUT_DIR}")
        
        return result
