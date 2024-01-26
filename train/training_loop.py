import copy
import functools
import os
import time
from types import SimpleNamespace
import numpy as np

import blobfile as bf
import torch
from torch.optim import AdamW

from diffusion import logger
from utils import dist_util
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler, UniformSampler
from tqdm import tqdm
from diffusion.resample import create_named_schedule_sampler
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from eval import eval_humanml, eval_humanact12_uestc
from data_loaders.get_data import get_dataset_loader
from itertools import cycle
from torch import nn
import data_loaders.humanml.utils.paramUtil as paramUtil
from diffusion.nn import mean_flat, sum_flat

from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml.utils.plot_script import plot_3d_array
import imageio
from utils.process_smpl_from_hybrik import amass_to_pose, pos2hmlrep
from data_loaders.tensors import collate


# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

loss_ce = nn.CrossEntropyLoss()
loss_mse = nn.MSELoss()
cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
loss_L1 = nn.L1Loss()

class TrainLoop:
    def __init__(self, args, train_platform, model, diffusion, data):
        self.args = args
        self.dataset = args.dataset
        self.train_platform = train_platform
        self.model = model
        self.diffusion = diffusion
        self.cond_mode = model.cond_mode
        self.data = data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1

        self.sync_cuda = torch.cuda.is_available()

        self.save_dir = args.save_dir
        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.overwrite = args.overwrite

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None
        if args.dataset in ['kit', 'humanml'] and args.eval_during_training:
            mm_num_samples = 0  # mm is super slow hence we won't run it during training
            mm_num_repeats = 0  # mm is super slow hence we won't run it during training
            gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
                                            split=args.eval_split,
                                            hml_mode='eval')

            self.eval_gt_data = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
                                                   split=args.eval_split,
                                                   hml_mode='gt')
            self.eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
            self.eval_data = {
                'test': lambda: eval_humanml.get_mdm_loader(
                    model, diffusion, args.eval_batch_size,
                    gen_loader, mm_num_samples, mm_num_repeats, gen_loader.dataset.opt.max_motion_length,
                    args.eval_num_samples, scale=1.,
                )
            }
        self.use_ddp = False
        self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        # resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        resume_checkpoint = find_resume_checkpoint(self.resume_checkpoint, 'model') if os.path.isdir(self.resume_checkpoint) else self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
            )

    def _load_optimizer_state(self):
        # main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        main_checkpoint = find_resume_checkpoint(self.resume_checkpoint, 'opt') if os.path.isdir(self.resume_checkpoint) else self.resume_checkpoint

        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):

        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}')
            for motion, cond in tqdm(self.data):
                if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                    break

                motion = motion.to(self.device)
                cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}

                self.run_step(motion, cond)
                if self.step % self.log_interval == 0:
                    for k,v in logger.get_current().name2val.items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(self.step+self.resume_step, v))

                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(name=k, value=v, iteration=self.step, group_name='Loss')

                if self.step % self.save_interval == 0:
                    self.save()
                    self.model.eval()
                    self.evaluate()
                    self.model.train()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1
            if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                break
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            self.evaluate()

    def evaluate(self):
        if not self.args.eval_during_training:
            return
        start_eval = time.time()
        if self.eval_wrapper is not None:
            print('Running evaluation loop: [Should take about 90 min]')
            log_file = os.path.join(self.save_dir, f'eval_humanml_{(self.step + self.resume_step):09d}.log')
            diversity_times = 300
            mm_num_times = 0  # mm is super slow hence we won't run it during training
            eval_dict = eval_humanml.evaluation(
                self.eval_wrapper, self.eval_gt_data, self.eval_data, log_file,
                replication_times=self.args.eval_rep_times, diversity_times=diversity_times, mm_num_times=mm_num_times, run_mm=False)
            print(eval_dict)
            for k, v in eval_dict.items():
                if k.startswith('R_precision'):
                    for i in range(len(v)):
                        self.train_platform.report_scalar(name=f'top{i + 1}_' + k, value=v[i],
                                                          iteration=self.step + self.resume_step,
                                                          group_name='Eval')
                else:
                    self.train_platform.report_scalar(name=k, value=v, iteration=self.step + self.resume_step,
                                                      group_name='Eval')

        elif self.dataset in ['humanact12', 'uestc']:
            eval_args = SimpleNamespace(num_seeds=self.args.eval_rep_times, num_samples=self.args.eval_num_samples,
                                        batch_size=self.args.eval_batch_size, device=self.device, guidance_param = 1,
                                        dataset=self.dataset, unconstrained=self.args.unconstrained,
                                        model_path=os.path.join(self.save_dir, self.ckpt_file_name()))
            eval_dict = eval_humanact12_uestc.evaluate(eval_args, model=self.model, diffusion=self.diffusion, data=self.data.dataset)
            print(f'Evaluation results on {self.dataset}: {sorted(eval_dict["feats"].items())}')
            for k, v in eval_dict["feats"].items():
                if 'unconstrained' not in k:
                    self.train_platform.report_scalar(name=k, value=np.array(v).astype(float).mean(), iteration=self.step, group_name='Eval')
                else:
                    self.train_platform.report_scalar(name=k, value=np.array(v).astype(float).mean(), iteration=self.step, group_name='Eval Unconstrained')

        end_eval = time.time()
        print(f'Evaluation time: {round(end_eval-start_eval)/60}min')


    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,  # [bs, ch, image_size, image_size]
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond,
                dataset=self.data.dataset
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)


    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"


    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights
            clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]
            for e in clip_weights:
                del state_dict[e]

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)


class TrainLoop_Style:
    def __init__(self, args, train_platform, model, diffusion, t2m_data, sty_data):
        self.args = args
        self.dataset = args.dataset
        self.train_platform = train_platform
        self.model = model
        self.diffusion = diffusion
        self.cond_mode = model.cond_mode
        self.t2m_data = t2m_data
        self.sty_data = sty_data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.t2m_data) + 1

        self.sync_cuda = torch.cuda.is_available()

        self.save_dir = args.save_dir
        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.overwrite = args.overwrite
        self.middle_trans = args.middle_trans

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        # if self.resume_step:
        #     self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None
        if args.dataset in ['kit', 'humanml'] and args.eval_during_training:
            mm_num_samples = 0  # mm is super slow hence we won't run it during training
            mm_num_repeats = 0  # mm is super slow hence we won't run it during training
            gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
                                            split=args.eval_split,
                                            hml_mode='eval')

            self.eval_gt_data = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
                                                   split=args.eval_split,
                                                   hml_mode='gt')
            self.eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
            self.eval_data = {
                'test': lambda: eval_humanml.get_mdm_loader(
                    model, diffusion, args.eval_batch_size,
                    gen_loader, mm_num_samples, mm_num_repeats, gen_loader.dataset.opt.max_motion_length,
                    args.eval_num_samples, scale=1.,
                )
            }
        self.use_ddp = False
        self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        # resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        resume_checkpoint = find_resume_checkpoint(self.resume_checkpoint, 'model') if os.path.isdir(self.resume_checkpoint) else self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            missing_keys, unexpected_keys = self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()), strict=False
            )
            assert len(unexpected_keys) == 0
            assert all([k.startswith('clip_model.') or 
                k.startswith('sty_enc.') or 
                k.startswith('adaIN.') for k in missing_keys])

    def _load_optimizer_state(self):
        # main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        main_checkpoint = find_resume_checkpoint(self.resume_checkpoint, 'opt') if os.path.isdir(self.resume_checkpoint) else self.resume_checkpoint

        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):

        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}')
            if len(self.sty_data.dataset) < len(self.t2m_data.dataset):
                self.sty_data_cycle = cycle(self.sty_data)
                self.t2m_data_cycle = self.t2m_data
            else:
                self.t2m_data_cycle = cycle(self.t2m_data)
                self.sty_data_cycle = self.sty_data
                   
             
            if np.random.rand() > 0.5:
                for motion, cond in tqdm(self.t2m_data):
            
                    if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                        break
                    
                    motion = motion.to(self.device)
                    cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}
                   
                    self.run_step(motion, cond)
                    if self.step % self.log_interval == 0:
                        for k,v in logger.get_current().name2val.items():
                            if k == 'loss':
                                print('step[{}]: loss[{:0.5f}]'.format(self.step+self.resume_step, v))

                            if k in ['step', 'samples'] or '_q' in k:
                                continue
                            else:
                                self.train_platform.report_scalar(name=k, value=v, iteration=self.step, group_name='Loss')

                    if self.step % self.save_interval == 0:
                        self.save()
                        self.model.eval()
                        self.evaluate()
                        self.model.train()

                        # Run for a finite amount of time in integration tests.
                        if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                            return
                    self.step += 1
            else:
                
                # style transfer finetuning 
                for t2m, sty in tqdm(zip(self.t2m_data_cycle, self.sty_data_cycle)):
                    motion, cond = t2m
                    sty_motion, sty_cond = sty
                    cond.update({"sty_x": sty_motion})
                    cond.update({"sty_y": sty_cond.pop("y")})
                    
                    if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                        break
                    
                    motion = motion.to(self.device)
                    cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}
                    cond['sty_x'] = cond['sty_x'].to(self.device)
                    cond['sty_y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['sty_y'].items()}
                    cond['sty_y'].update({"middle_trans": self.middle_trans})

                    self.run_step(motion, cond)
                    if self.step % self.log_interval == 0:
                        for k,v in logger.get_current().name2val.items():
                            if k == 'loss':
                                print('step[{}]: loss[{:0.5f}]'.format(self.step+self.resume_step, v))

                            if k in ['step', 'samples'] or '_q' in k:
                                continue
                            else:
                                self.train_platform.report_scalar(name=k, value=v, iteration=self.step, group_name='Loss')

                    if self.step % self.save_interval == 0:
                        self.save()
                        self.model.eval()
                        self.evaluate()
                        self.model.train()

                        # Run for a finite amount of time in integration tests.
                        if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                            return
                    self.step += 1
            if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                break
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            self.evaluate()

    def evaluate(self):
        if not self.args.eval_during_training:
            return
        start_eval = time.time()
        if self.eval_wrapper is not None:
            print('Running evaluation loop: [Should take about 90 min]')
            log_file = os.path.join(self.save_dir, f'eval_humanml_{(self.step + self.resume_step):09d}.log')
            diversity_times = 300
            mm_num_times = 0  # mm is super slow hence we won't run it during training
            eval_dict = eval_humanml.evaluation(
                self.eval_wrapper, self.eval_gt_data, self.eval_data, log_file,
                replication_times=self.args.eval_rep_times, diversity_times=diversity_times, mm_num_times=mm_num_times, run_mm=False)
            print(eval_dict)
            for k, v in eval_dict.items():
                if k.startswith('R_precision'):
                    for i in range(len(v)):
                        self.train_platform.report_scalar(name=f'top{i + 1}_' + k, value=v[i],
                                                          iteration=self.step + self.resume_step,
                                                          group_name='Eval')
                else:
                    self.train_platform.report_scalar(name=k, value=v, iteration=self.step + self.resume_step,
                                                      group_name='Eval')

        elif self.dataset in ['humanact12', 'uestc']:
            eval_args = SimpleNamespace(num_seeds=self.args.eval_rep_times, num_samples=self.args.eval_num_samples,
                                        batch_size=self.args.eval_batch_size, device=self.device, guidance_param = 1,
                                        dataset=self.dataset, unconstrained=self.args.unconstrained,
                                        model_path=os.path.join(self.save_dir, self.ckpt_file_name()))
            eval_dict = eval_humanact12_uestc.evaluate(eval_args, model=self.model, diffusion=self.diffusion, data=self.data.dataset)
            print(f'Evaluation results on {self.dataset}: {sorted(eval_dict["feats"].items())}')
            for k, v in eval_dict["feats"].items():
                if 'unconstrained' not in k:
                    self.train_platform.report_scalar(name=k, value=np.array(v).astype(float).mean(), iteration=self.step, group_name='Eval')
                else:
                    self.train_platform.report_scalar(name=k, value=np.array(v).astype(float).mean(), iteration=self.step, group_name='Eval Unconstrained')

        end_eval = time.time()
        print(f'Evaluation time: {round(end_eval-start_eval)/60}min')


    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,  # [bs, ch, image_size, image_size]
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond,
                dataset=self.t2m_data.dataset
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)


    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"


    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights
            clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]
            for e in clip_weights:
                del state_dict[e]

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)


class TrainLoopMotionEncoder:
    def __init__(self, args, train_platform, model, data, t2m_data=None, diffusion=None):
        self.args = args
        self.dataset = args.dataset
        self.train_platform = train_platform
        self.model = model
        self.data = data
        self.t2m_data = t2m_data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1

        self.sync_cuda = torch.cuda.is_available()

        self.save_dir = args.save_dir
        self._load_and_sync_parameters()
        # for p in self.model.input_process.parameters():
        #     print(p.requires_grad)
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.overwrite = args.overwrite
        
        self.diffusion = diffusion
        if diffusion is not None:
            self.schedule_sampler_type = 'uniform'
            self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        
    def _load_and_sync_parameters(self):
        # resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        resume_checkpoint = find_resume_checkpoint(self.resume_checkpoint, 'model') if os.path.isdir(self.resume_checkpoint) else self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            missing_keys, unexpected_keys = self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                ), strict=False
            )
            # assert len(unexpected_keys) == 0
            assert all([k.startswith('input_process.') for k in unexpected_keys])
            assert all([k.startswith('clip_model.') or 
                        k.startswith('mdm_model.') for k in missing_keys])
            

    def _load_optimizer_state(self):
        # main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        main_checkpoint = find_resume_checkpoint(self.resume_checkpoint, 'opt') if os.path.isdir(self.resume_checkpoint) else self.resume_checkpoint

        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            try:
                self.opt.load_state_dict(state_dict)
            except:
                pass

    def run_loop(self):

        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}')
            
            if self.t2m_data is not None:
                if len(self.data.dataset) < len(self.t2m_data.dataset):
                    self.data_cycle = cycle(self.data)
                    self.t2m_data_cycle = self.t2m_data
                else:
                    self.t2m_data_cycle = cycle(self.t2m_data)
                    self.data_cycle = self.data   
            if self.t2m_data is not None:
                zip_motion = zip(self.data_cycle, self.t2m_data_cycle)
            else:
                zip_motion = self.data
                
            for arxiv, t2m_arxiv in tqdm(zip_motion):
                
                if self.t2m_data is not None:
                    motion, cond = arxiv
                    t2m_motion, t2m_cond = t2m_arxiv
                else:
                    motion = arxiv
                    cond = t2m_arxiv  
                    t2m_motion = None
                    t2m_cond = None
                if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                    break

                motion = motion.to(self.device)
                cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}
                if self.t2m_data is not None:
                    t2m_motion = t2m_motion.to(self.device)
                    t2m_cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in t2m_cond['y'].items()}

                self.run_step(motion, cond, t2m_motion, t2m_cond)
                if self.step % self.log_interval == 0:
                    for k,v in logger.get_current().name2val.items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(self.step+self.resume_step, v))

                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(name=k, value=v, iteration=self.step, group_name='Loss')

                if self.step % self.save_interval == 0:
                    self.save()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1
            if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                break
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond, t2m_batch=None, t2m_cond=None):
        self.forward_backward(batch, cond, t2m_batch, t2m_cond)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    def training_losses(self, model, batch, model_kwargs=None):

        terms = {}
    
        mu, text_features = model(batch, **model_kwargs)  
        features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        mu_norm = mu / mu.norm(dim=-1, keepdim=True)

        cos = cosine_sim(features_norm, mu_norm)
        cosine_loss = (1 - cos).mean()
        terms["text_cosine"] = cosine_loss

        terms["loss"] = cosine_loss
        
        return terms

    def forward_backward(self, batch, cond, t2m_batch=None, t2m_cond=None):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            micro_t2m = t2m_batch
            micro_cond_t2m = t2m_cond
            # last_batch = (i + self.microbatch) >= batch.shape[0]
            
            if self.diffusion is None:
                compute_losses = functools.partial(
                    self.training_losses,
                    self.model,
                    micro,
                    model_kwargs=micro_cond
                )
            else:
                t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
                micro = self.model.mdm_model.re_encode(micro).detach()
                compute_losses = functools.partial(
                    self.diffusion.finetune_motionenc_losses,
                    self.model,
                    micro,
                    t,
                    model_kwargs=micro_cond
                )

                compute_t2m_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.model.mdm_model,
                    micro_t2m,
                    t,
                    model_kwargs = micro_cond_t2m,
                    dataset =self.t2m_data.dataset
                )
                
                compute_text_losses = functools.partial(
                    self.training_losses,
                    self.model,
                    micro_t2m,
                    model_kwargs=micro_cond_t2m
                )


            losses = compute_losses()
            if self.diffusion is not None:
                t2m_losses = compute_t2m_losses()
                text_losses = compute_text_losses()
                losses["loss"] += t2m_losses.pop("loss")
                losses["loss"] += text_losses.pop("loss")

            if self.diffusion is not None:
                if isinstance(self.schedule_sampler, LossAwareSampler):
                    self.schedule_sampler.update_with_local_losses(
                        t, losses["loss"].detach()
                    )
                loss = (losses["loss"] * weights).mean()
                log_motion_encoder_finetune_loss_dict(
                    {k: v for k, v in losses.items()}
                )
                log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in t2m_losses.items()}
                )
                log_motion_encoder_loss_dict(
                    {k: v for k, v in text_losses.items()}
                )
            else:
                loss = (losses["loss"])
                log_motion_encoder_loss_dict(
                    {k: v for k, v in losses.items()}
                )
            self.mp_trainer.backward(loss)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"

    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights
            clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]
            # mdm_weights = [e for e in state_dict.keys() if e.startswith('mdm_model.')]
            for e in clip_weights:
                del state_dict[e]
            
            # for e in mdm_weights:
            #     del state_dict[e]
            if self.dataset != 'humanml':
                mdm_weights = [e for e in state_dict.keys() if e.startswith('mdm_model.')]
                for e in mdm_weights:
                    del state_dict[e]
            

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)


class TrainAELoop(TrainLoopMotionEncoder):
    def __init__(self, args, train_platform, model, data, t2m_data=None, diffusion=None):
        super().__init__(args, train_platform, model, data, t2m_data, diffusion)
    
        self.l2_loss = lambda a, b: (a - b) ** 2 

    def masked_l2(self, a, b, mask):
        # assuming a.shape == b.shape == bs, J, Jdim, seqlen
        # assuming mask.shape == bs, 1, 1, seqlen
        loss = self.l2_loss(a, b)
        loss = sum_flat(loss * mask.float())  # gives \sigma_euclidean over unmasked elements
        n_entries = a.shape[1] * a.shape[2]
        non_zero_elements = sum_flat(mask) * n_entries
        # print('mask', mask.shape)
        # print('non_zero_elements', non_zero_elements)
        # print('loss', loss)
        mse_loss_val = loss / non_zero_elements
        # print('mse_loss_val', mse_loss_val)
        return mse_loss_val    

    def training_losses(self, model, batch, model_kwargs=None):
        mask = model_kwargs['y']['mask']
        
        terms = {}
        rec = model(batch) 
        # loss =self.loss_mse(rec, batch)
        rec_loss = self.masked_l2(batch, rec, mask).mean()
        terms["loss"] = rec_loss # mean_flat(rot_mse)

        return terms

    
    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)


class TrainDisLoop(TrainLoopMotionEncoder):
    def __init__(self, args, train_platform, model, data, t2m_data=None, diffusion=None):
        super().__init__(args, train_platform, model, data, t2m_data, diffusion)
    
        self.l2_loss = lambda a, b: (a - b) ** 2 
        
        self.style_id_mapping = {
                "angry": 0,
                "childlike":1,
                "depressed":2,
                "neutral":3,
                "old":4,
                "proud":5,
                "sexy":6,
                "strutting":7
            }
        

    def masked_l2(self, a, b, mask):
        # assuming a.shape == b.shape == bs, J, Jdim, seqlen
        # assuming mask.shape == bs, 1, 1, seqlen
        loss = self.l2_loss(a, b)
        loss = sum_flat(loss * mask.float())  # gives \sigma_euclidean over unmasked elements
        n_entries = a.shape[1] * a.shape[2]
        non_zero_elements = sum_flat(mask) * n_entries
        # print('mask', mask.shape)
        # print('non_zero_elements', non_zero_elements)
        # print('loss', loss)
        mse_loss_val = loss / non_zero_elements
        # print('mse_loss_val', mse_loss_val)
        return mse_loss_val    

    def training_losses(self, model, batch, model_kwargs=None):
        mask = model_kwargs['y']['mask']
        texts = model_kwargs['y']['text']
        style_id = []
        for text in texts:
            style_name = text.split(" ")[-1]
            style_id.append(self.style_id_mapping[style_name])
        style_id = torch.Tensor(style_id).long().to(batch.device)
            
        terms = {}
        real_loss, real_accuracy, resp_real = model.calc_dis_real_loss(batch, style_id) 
        
        terms["real_accuracy"] = real_accuracy
        terms["loss"] = real_loss # mean_flat(rot_mse)
        
        return terms

    
    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)


class TrainLoopTrajectory(TrainLoopMotionEncoder):
    def __init__(self, args, train_platform, model, data, t2m_data=None, diffusion=None):
        super().__init__(args, train_platform, model, data, t2m_data, diffusion)

        assert t2m_data is None
        assert diffusion is None
        self.l2_loss = lambda a, b: (a - b) ** 2

    def masked_l2(self, a, b, mask):
        # assuming a.shape == b.shape == bs, J, Jdim, seqlen
        # assuming mask.shape == bs, 1, 1, seqlen
        loss = self.l2_loss(a, b)
        loss = sum_flat(loss * mask.float())  # gives \sigma_euclidean over unmasked elements
        n_entries = a.shape[1] * a.shape[2]
        non_zero_elements = sum_flat(mask) * n_entries
        # print('mask', mask.shape)
        # print('non_zero_elements', non_zero_elements)
        # print('loss', loss)
        mse_loss_val = loss / non_zero_elements
        # print('mse_loss_val', mse_loss_val)
        return mse_loss_val    

    def training_losses(self, model, batch, model_kwargs=None):

        terms = {}
        mask = model_kwargs['y']['mask']

        motion_trajectory = model(batch, **model_kwargs)  # [bs, 4, 1, nframes]
        
        gt_trajectory = batch[:, :4, ...]

        terms["rot_mse"] = self.masked_l2(gt_trajectory, motion_trajectory, mask).mean() # mean_flat(rot_mse)
        
        terms["loss"] = terms["rot_mse"]
        
        return terms
    
    def forward_backward(self, batch, cond, t2m_batch=None, t2m_cond=None):
        assert t2m_batch is None
        assert t2m_cond is None
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            # last_batch = (i + self.microbatch) >= batch.shape[0]
            
            compute_losses = functools.partial(
                self.training_losses,
                self.model,
                micro,
                model_kwargs=micro_cond
            )

            losses = compute_losses()
            
            loss = (losses["loss"])
            log_motion_encoder_loss_dict(
                {k: v for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)


            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)

        self.model = self.model.eval()
        motion_dir = './videos/res_xixiyu/'
        path = os.path.join(motion_dir, 'res.pk')
        fps_video = 30
        start_second = 52.4 
        end_second = 55.5

        pose_seq_np_n, joints = amass_to_pose(path, fps_video, trans_path="", with_trans=False)
        path = pos2hmlrep(joints)
        path = path[int(20*start_second):int(20*end_second)]
        
        t2m_motion, m_length = self.data.dataset.t2m_dataset.process_np_motion(path)
        t2m_motion = torch.Tensor(t2m_motion.T).unsqueeze(1).unsqueeze(0)
        t2m_motion = t2m_motion.to(dist_util.dev()) 

        collate_args = [{'inp': torch.zeros(196), 'tokens': None, 'lengths': m_length}] * 1
   
        texts = ["input_motion"] * 1

        collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
        _, model_kwargs = collate(collate_args)
        model_kwargs["y"] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs["y"].items()}

        with torch.no_grad():
            pred_root = self.model(t2m_motion, **model_kwargs)
    
        pred_motion = torch.cat((pred_root, t2m_motion[:, 4:, ...]), 1)

        n_joints = 22 
        
        sample_predict = self.data.dataset.t2m_dataset.inv_transform(pred_motion.detach().cpu().permute(0, 2, 3, 1)).float()
        sample_predict = recover_from_ric(sample_predict, n_joints) # B 1 T J 3 
        sample_predict = sample_predict.view(-1, *sample_predict.shape[2:]).permute(0, 2, 3, 1)  # B J 3 T 

        
        caption = model_kwargs['y']['text']
        
        sample_predict = sample_predict.cpu().numpy().transpose(0, 3, 1, 2)
    
        sample_array = plot_3d_array([sample_predict[0][:model_kwargs["y"]["lengths"][0]], None, paramUtil.t2m_kinematic_chain, caption[0] + "_predict_trajectory"])
        imageio.mimsave(os.path.join(self.save_dir, f'pred_root_trajectory_{(self.step+self.resume_step):09d}.gif'), np.array(sample_array), duration=int(model_kwargs["y"]["lengths"][0]/20))

        self.model = self.model.train()
    

class TrainLoopTransferModule:
    def __init__(self, args, train_platform, model, data, t2m_data=None, diffusion=None):
        self.args = args
        self.dataset = args.dataset
        self.train_platform = train_platform
        self.model = model
        self.data = data
        self.t2m_data = t2m_data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1

        self.sync_cuda = torch.cuda.is_available()

        self.save_dir = args.save_dir
        self._load_and_sync_parameters()
        # for p in self.model.input_process.parameters():
        #     print(p.requires_grad)
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.overwrite = args.overwrite

        self.l2_loss = lambda a, b: (a - b) ** 2 
        
        self.diffusion = diffusion
        if diffusion is not None:
            self.schedule_sampler_type = 'uniform'
            self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        
    def _load_and_sync_parameters(self):
        # resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        resume_checkpoint = find_resume_checkpoint(self.resume_checkpoint, 'model') if os.path.isdir(self.resume_checkpoint) else self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            missing_keys, unexpected_keys = self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                ), strict=False
            )
            assert len(unexpected_keys) == 0
            assert all([k.startswith("motion_enc.") for k in missing_keys])
    

    def _load_optimizer_state(self):
        # main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        main_checkpoint = find_resume_checkpoint(self.resume_checkpoint, 'opt') if os.path.isdir(self.resume_checkpoint) else self.resume_checkpoint

        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            try:
                self.opt.load_state_dict(state_dict)
            except:
                pass

    def run_loop(self):

        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}')
            
            if self.t2m_data is not None:
                if len(self.data.dataset) < len(self.t2m_data.dataset):
                    self.data_cycle = cycle(self.data)
                    self.t2m_data_cycle = self.t2m_data
                else:
                    self.t2m_data_cycle = cycle(self.t2m_data)
                    self.data_cycle = self.data   
            if self.t2m_data is not None:
                zip_motion = zip(self.data_cycle, self.t2m_data_cycle)
            else:
                zip_motion = self.data
                
            for arxiv, t2m_arxiv in tqdm(zip_motion):
                
                if self.t2m_data is not None:
                    (motion, cond), (normal_motion, normal_cond) = arxiv
                    t2m_motion, t2m_cond = t2m_arxiv
                else:
                    motion, cond = arxiv
                    normal_motion, normal_cond = t2m_arxiv  
                    t2m_motion = None
                    t2m_cond = None
                if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                    break

                motion = motion.to(self.device)
                cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}

                normal_motion = normal_motion.to(self.device)
                normal_cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in normal_cond['y'].items()}
                if self.t2m_data is not None:
                    t2m_motion = t2m_motion.to(self.device)
                    t2m_cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in t2m_cond['y'].items()}

                self.run_step(motion, cond, normal_motion, normal_cond, t2m_motion, t2m_cond)
                if self.step % self.log_interval == 0:
                    for k,v in logger.get_current().name2val.items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(self.step+self.resume_step, v))

                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(name=k, value=v, iteration=self.step, group_name='Loss')
                        # for name, param in self.model.named_parameters():
                        #     if param.requires_grad and name == 'control_seqTransEncoder.layers.1.norm1.weight':
                        #         print(name, param.grad)
                        # print("******************")


                if self.step % self.save_interval == 0:
                    self.save()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1
            if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                break
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond, normal_batch, normal_cond, t2m_batch=None, t2m_cond=None):
        self.forward_backward(batch, cond, normal_batch, normal_cond, t2m_batch, t2m_cond)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    def training_losses(self, model, batch, batch_normal, t2m_batch, kwargs=None, normal_kwargs=None, t2m_kwargs=None):

        terms = {}
        
        style_mu, _ = model.motion_enc(batch, **kwargs)
        normal_mu, _ = model.motion_enc(batch_normal, **normal_kwargs)
        t2m_mu, _ = model.motion_enc(t2m_batch, **t2m_kwargs)
        transferred_motion = model(t2m_batch, normal_mu, style_mu) 
        
        style = style_mu - normal_mu
        transferred_mu, _ = model.motion_enc(transferred_motion, **t2m_kwargs)
        
        trans_style = transferred_mu - t2m_mu

        style_norm = style / (style.norm(dim=-1, keepdim=True) + 1e-6)
        trans_style_norm = trans_style / (trans_style.norm(dim=-1, keepdim=True) + 1e-6)

        # cos = cosine_sim(style, trans_style) This is buggy
        cos = cosine_sim(style_norm, trans_style_norm)
        cosine_loss = (1 - cos).mean()
        terms["style_cosine"] = cosine_loss
        
        # l1_loss = loss_L1(style, trans_style).mean()
        l1_loss = loss_mse(style, trans_style).mean()
        terms["style_l1"] = l1_loss
        
        
        transferred_bones = self.calculate_bone(transferred_motion).squeeze().permute(0,2,3,1) # B J-1 3,seq
        source_bones = self.calculate_bone(t2m_batch).squeeze().permute(0,2,3,1)
        bone_regularization = self.masked_l2(transferred_bones, source_bones, t2m_kwargs["y"]["mask"]).mean()
        terms["bone_regularization"] = bone_regularization
        
        terms["loss"] = cosine_loss + self.args.lambda_l1 * l1_loss + self.args.lambda_bone * bone_regularization
        # terms["loss"] = self.args.lambda_l1 * l1_loss + self.args.lambda_bone * bone_regularization
        
        return terms
    
    def calculate_bone(self, motion):
        ### motion is normalized motion B J 1 T
        joint_num = 22 if self.args.dataset == 'humanml' else 23
        denorm_motion = self.t2m_data.dataset.t2m_dataset.inv_transform_tensor(motion.permute(0, 2, 3, 1))
        # B 1 T J
        vel = denorm_motion[..., :4]  
        ric_data = denorm_motion[..., 4 : 4 + (joint_num - 1) * 3] 
        ric_data = ric_data.reshape(ric_data.shape[:-1] + ((joint_num - 1), 3)) # x,z are relative to root joint, all face z+
        root_ric = torch.zeros_like(ric_data[..., 0:1, :]).to(ric_data.device)
        root_ric[...,0, 1] = vel[..., 3] 
        ric_data = torch.cat((root_ric, ric_data), dim=-2)

        chains = paramUtil.t2m_kinematic_chain
        bones = []
        for chain in chains:
            for i in range(1, len(chain)):
                bones.append((ric_data[..., chain[i], :] - ric_data[..., chain[i-1], :]).unsqueeze(-2))
        bones = torch.cat(bones, -2)
        return bones              # B 1 T J-1 3
    

    def masked_l2(self, a, b, mask):
        # assuming a.shape == b.shape == bs, J, Jdim, seqlen
        # assuming mask.shape == bs, 1, 1, seqlen
        
        loss = self.l2_loss(a, b)
        loss = sum_flat(loss * mask.float())  # gives \sigma_euclidean over unmasked elements
        n_entries = a.shape[1] * a.shape[2]
        non_zero_elements = sum_flat(mask) * n_entries
        # print('mask', mask.shape)
        # print('non_zero_elements', non_zero_elements)
        # print('loss', loss)
        mse_loss_val = loss / non_zero_elements
        # print('mse_loss_val', mse_loss_val)
        return mse_loss_val
    

    def forward_backward(self, batch, cond, normal_batch, normal_cond, t2m_batch=None, t2m_cond=None):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            micro_normal = normal_batch
            micro_normal_cond = normal_cond
            micro_t2m = t2m_batch
            micro_cond_t2m = t2m_cond
            # last_batch = (i + self.microbatch) >= batch.shape[0]
            
            if self.diffusion is None:
                compute_losses = functools.partial(
                    self.training_losses,
                    self.model,
                    micro,
                    micro_normal,
                    micro_t2m,
                    kwargs=micro_cond,
                    normal_kwargs=micro_normal_cond,
                    t2m_kwargs=micro_cond_t2m
                )


            losses = compute_losses()
        
            loss = losses["loss"]
            log_motion_encoder_loss_dict(
                {k: v for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)
            # for name, param in self.model.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.grad)
            # print("******************")

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"

    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights
            clip_weights = [e for e in state_dict.keys() if e.startswith('motion_enc.')]
            # mdm_weights = [e for e in state_dict.keys() if e.startswith('mdm_model.')]
            for e in clip_weights:
                del state_dict[e]
            
            # for e in mdm_weights:
            #     del state_dict[e]

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)


class TrainLoopStyleDiffusion:
    def __init__(self, args, train_platform, model, data, t2m_data=None, diffusion=None):
        self.args = args
        self.dataset = args.dataset
        self.train_platform = train_platform
        self.model = model
        self.data = data
        self.t2m_data = t2m_data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1

        self.sync_cuda = torch.cuda.is_available()

        self.save_dir = args.save_dir
        self._load_and_sync_parameters()

        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.overwrite = args.overwrite

        self.l2_loss = lambda a, b: (a - b) ** 2 
        
        self.diffusion = diffusion
        if diffusion is not None:
            self.schedule_sampler_type = 'uniform'
            self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        
    def _load_and_sync_parameters(self):
        # resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        resume_checkpoint = find_resume_checkpoint(self.resume_checkpoint, 'model') if os.path.isdir(self.resume_checkpoint) else self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            missing_keys, unexpected_keys = self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                ), strict=False
            )
            assert len(unexpected_keys) == 0
            assert all([k.startswith("motion_enc.") for k in missing_keys])
    

    def _load_optimizer_state(self):
        # main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        main_checkpoint = find_resume_checkpoint(self.resume_checkpoint, 'opt') if os.path.isdir(self.resume_checkpoint) else self.resume_checkpoint

        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            try:
                self.opt.load_state_dict(state_dict)
            except:
                pass

    def run_loop(self):

        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}')
            
            if self.t2m_data is not None:
                if len(self.data.dataset) < len(self.t2m_data.dataset):
                    self.data_cycle = cycle(self.data)
                    self.t2m_data_cycle = self.t2m_data
                else:
                    self.t2m_data_cycle = cycle(self.t2m_data)
                    self.data_cycle = self.data   
            if self.t2m_data is not None:
                zip_motion = zip(self.data_cycle, self.t2m_data_cycle)
            else:
                zip_motion = self.data
                
            for arxiv, t2m_arxiv in tqdm(zip_motion):
                
                if self.t2m_data is not None:
                    (motion, cond), (normal_motion, normal_cond) = arxiv
                    t2m_motion, t2m_cond = t2m_arxiv
                else:
                    motion, cond = arxiv
                    normal_motion, normal_cond = t2m_arxiv  
                    t2m_motion = None
                    t2m_cond = None
                if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                    break

                motion = motion.to(self.device)
                cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}

                normal_motion = normal_motion.to(self.device)
                normal_cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in normal_cond['y'].items()}
                if self.t2m_data is not None:
                    t2m_motion = t2m_motion.to(self.device)
                    t2m_cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in t2m_cond['y'].items()}

                self.run_step(motion, cond, normal_motion, normal_cond, t2m_motion, t2m_cond)
                if self.step % self.log_interval == 0:
                    for k,v in logger.get_current().name2val.items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(self.step+self.resume_step, v))

                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(name=k, value=v, iteration=self.step, group_name='Loss')
                        # for name, param in self.model.named_parameters():
                        #     if param.requires_grad and name == 'control_seqTransEncoder.layers.1.norm1.weight':
                        #         print(name, param.grad)
                        # print("******************")


                if self.step % self.save_interval == 0:
                    self.save()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1
            if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                break
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond, normal_batch, normal_cond, t2m_batch=None, t2m_cond=None):
        self.forward_backward(batch, cond, normal_batch, normal_cond, t2m_batch, t2m_cond)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    
    def calculate_bone(self, motion):
        ### motion is normalized motion B J 1 T
        joint_num = 22 if self.args.dataset == 'humanml' else 23
        denorm_motion = self.t2m_data.dataset.t2m_dataset.inv_transform_tensor(motion.permute(0, 2, 3, 1))
        # B 1 T J
        vel = denorm_motion[..., :4]  
        ric_data = denorm_motion[..., 4 : 4 + (joint_num - 1) * 3] 
        ric_data = ric_data.reshape(ric_data.shape[:-1] + ((joint_num - 1), 3)) # x,z are relative to root joint, all face z+
        root_ric = torch.zeros_like(ric_data[..., 0:1, :]).to(ric_data.device)
        root_ric[...,0, 1] = vel[..., 3] 
        ric_data = torch.cat((root_ric, ric_data), dim=-2)

        chains = paramUtil.t2m_kinematic_chain
        bones = []
        for chain in chains:
            for i in range(1, len(chain)):
                bones.append((ric_data[..., chain[i], :] - ric_data[..., chain[i-1], :]).unsqueeze(-2))
        bones = torch.cat(bones, -2)
        return bones              # B 1 T J-1 3

    def forward_backward(self, batch, cond, normal_batch, normal_cond, t2m_batch=None, t2m_cond=None):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            micro_normal = normal_batch
            micro_normal_cond = normal_cond
            micro_t2m = t2m_batch
            micro_cond_t2m = t2m_cond
            # last_batch = (i + self.microbatch) >= batch.shape[0]
            
            assert self.diffusion is not None

            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.model,
                micro_t2m,
                t,
                model_kwargs = micro_cond_t2m,
                batch_normal = micro_normal,
                normal_kwargs = micro_normal_cond,
                batch_style = micro,
                style_kwargs = micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
                )

            self.mp_trainer.backward(loss)
            # for name, param in self.model.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.grad)
            # print("******************")

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"

    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights
            clip_weights = [e for e in state_dict.keys() if e.startswith('motion_enc.')]
            # mdm_weights = [e for e in state_dict.keys() if e.startswith('mdm_model.')]
            for e in clip_weights:
                del state_dict[e]
            
            # for e in mdm_weights:
            #     del state_dict[e]

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)


class TrainInpaintingLoop:
    def __init__(self, args, train_platform, model, data, diffusion=None, style_data=None):
        self.args = args
        self.dataset = args.dataset
        self.train_platform = train_platform
        self.model = model
        self.data = data
        self.style_data = style_data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps
        
        self.style_finetune = args.style_finetune if hasattr(args, "style_finetune") else 0
        self.weakly_style_pair = args.weakly_style_pair if hasattr(args, "style_finetune") else 0 

        self.skip_steps = args.skip_steps if hasattr(args, "skip_steps") else 0
        if hasattr(args, "skip_steps"):
            self.style_example = True
        else:
            self.style_example = False

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1

        self.sync_cuda = torch.cuda.is_available()

        self.save_dir = args.save_dir
        self._load_and_sync_parameters()

        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.overwrite = args.overwrite

        self.l2_loss = lambda a, b: (a - b) ** 2 
        
        self.diffusion = diffusion
        if diffusion is not None:
            self.schedule_sampler_type = 'uniform'
            self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        
    def _load_and_sync_parameters(self):
        # resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        resume_checkpoint = find_resume_checkpoint(self.resume_checkpoint, 'model') if os.path.isdir(self.resume_checkpoint) else self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            missing_keys, unexpected_keys = self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                ), strict=False
            )
            assert len(unexpected_keys) == 0
            assert all([k.startswith("motion_enc.") for k in missing_keys])
    

    def _load_optimizer_state(self):
        # main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        main_checkpoint = find_resume_checkpoint(self.resume_checkpoint, 'opt') if os.path.isdir(self.resume_checkpoint) else self.resume_checkpoint

        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            try:
                self.opt.load_state_dict(state_dict)
            except:
                pass

    def run_loop(self):
        if self.style_finetune:
            iter_styledata = iter(self.style_data)
        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}')
        
            if self.style_finetune:
                try:
                    content_motion, cond_style = next(iter_styledata)
                except:
                    iter_styledata = iter(self.style_data)
                    content_motion, cond_style = next(iter_styledata)
                cond_style['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond_style['y'].items()}
            else:
                content_motion = None
                cond_style = None
            
            for motion, cond in tqdm(self.data):
                if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                    break

                motion = motion.to(self.device)
                cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}
                
                self.run_step(motion, cond, content_motion, cond_style)
                if self.step % self.log_interval == 0:
                    for k,v in logger.get_current().name2val.items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(self.step+self.resume_step, v))

                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(name=k, value=v, iteration=self.step, group_name='Loss')
                        # for name, param in self.model.named_parameters():
                        #     if param.requires_grad and name == 'control_seqTransEncoder.layers.1.norm1.weight':
                        #         print(name, param.grad)
                        # print("******************")


                if self.step % self.save_interval == 0:
                    self.save()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1
            if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                break
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond, style_batch=None, style_cond=None):
        self.forward_backward(batch, cond, style_batch, style_cond)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    
    def calculate_bone(self, motion):
        ### motion is normalized motion B J 1 T
        joint_num = 22 if self.args.dataset == 'humanml' else 23
        denorm_motion = self.t2m_data.dataset.t2m_dataset.inv_transform_tensor(motion.permute(0, 2, 3, 1))
        # B 1 T J
        vel = denorm_motion[..., :4]  
        ric_data = denorm_motion[..., 4 : 4 + (joint_num - 1) * 3] 
        ric_data = ric_data.reshape(ric_data.shape[:-1] + ((joint_num - 1), 3)) # x,z are relative to root joint, all face z+
        root_ric = torch.zeros_like(ric_data[..., 0:1, :]).to(ric_data.device)
        root_ric[...,0, 1] = vel[..., 3] 
        ric_data = torch.cat((root_ric, ric_data), dim=-2)

        chains = paramUtil.t2m_kinematic_chain
        bones = []
        for chain in chains:
            for i in range(1, len(chain)):
                bones.append((ric_data[..., chain[i], :] - ric_data[..., chain[i-1], :]).unsqueeze(-2))
        bones = torch.cat(bones, -2)
        return bones              # B 1 T J-1 3

    def forward_backward(self, batch, cond, style_batch, style_cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            micro_content = style_batch
            micro_style_cond = style_cond

            # last_batch = (i + self.microbatch) >= batch.shape[0]
            
            assert self.diffusion is not None

            last_batch = (i + self.microbatch) >= batch.shape[0]
            
            if self.style_finetune:
                if self.args.use_ddim:
                    t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev(), range(int((self.args.diffusion_steps-self.args.skip_steps)/self.args.diffusion_steps * 20)))
                else:
                    t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev(), range(self.args.diffusion_steps-self.args.skip_steps))
            else:
                t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            
            if self.style_finetune: 
                compute_losses = functools.partial(
                    self.diffusion.few_shot_trans_losses,
                    self.model,
                    micro,
                    t,
                    micro_content,
                    micro_style_cond["y"]["inpainted_motion"],
                    skip_steps=self.args.skip_steps,
                    model_kwargs = micro_style_cond,
                    model_t2m_kwargs = micro_cond,
                    weakly_style_pair = self.weakly_style_pair,
                    use_ddim=self.args.use_ddim,
                    Ls = self.args.Ls
                    
                )
            else:
                compute_losses = functools.partial(
                    self.diffusion.training_inpainting_losses,
                    self.model,
                    micro,
                    t,
                    model_kwargs = micro_cond,
                    dataset=self.data.dataset
                )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
            
            if style_batch is None:
                loss = (losses["loss"] * weights).mean()
            else:
                loss = losses["loss"]
            
            if style_batch is None:
                log_loss_dict(
                    self.diffusion, t, {k: v * weights for k, v in losses.items()}
                    )
            else:
                log_loss_dict_style(
                    self.diffusion, t, losses
                )

            self.mp_trainer.backward(loss)
            # for name, param in self.model.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.grad)
            # print("******************")

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"

    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights
            if self.dataset != 'humanml':
                motion_enc_weights = [e for e in state_dict.keys() if e.startswith('motion_enc.')]
                clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]

                for e in motion_enc_weights:
                    del state_dict[e]

                for e in clip_weights:
                    del state_dict[e]

            else:
                controlmdm_weights = [e for e in state_dict.keys() if e.startswith('controlmdm.')]
                clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]
                # mdm_weights = [e for e in state_dict.keys() if e.startswith('mdm_model.')]
                for e in controlmdm_weights:
                    del state_dict[e]

                for e in clip_weights:
                    del state_dict[e]

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)


class FinetuneAELoop(TrainInpaintingLoop):
        def _load_optimizer_state(self):
            pass

        def save(self):
            pass
        
        
        def masked_l2(self, a, b, mask):
            # assuming a.shape == b.shape == bs, J, Jdim, seqlen
            # assuming mask.shape == bs, 1, 1, seqlen
            loss = self.l2_loss(a, b)
            loss = sum_flat(loss * mask.float())  # gives \sigma_euclidean over unmasked elements
            n_entries = a.shape[1] * a.shape[2]
            non_zero_elements = sum_flat(mask) * n_entries
            # print('mask', mask.shape)
            # print('non_zero_elements', non_zero_elements)
            # print('loss', loss)
            mse_loss_val = loss / non_zero_elements
            # print('mse_loss_val', mse_loss_val)
            return mse_loss_val    
        
        def ae_transfer_loss(self, model, x_start, t, x_content_start, x_style_start, skip_steps=700, model_kwargs=None, noise=None, model_t2m_kwargs=None, weakly_style_pair=0, use_ddim=0):
            try:
                motion_enc = model.controlmdm.motion_enc
            except:
                motion_enc = model.motion_enc
            mask = model_kwargs['y']['mask']
            mask_t2m = model_t2m_kwargs['y']['mask']

            model_output = model(x_start, None, None) 
            
            if weakly_style_pair:
                mu, text_features = motion_enc(model_output, **model_t2m_kwargs)

            sample = model(x_content_start,  None, None) # B J 1 seq
            terms = {}
     
            target = x_style_start

            target_t2m = x_start
            
            assert target.shape == x_content_start.shape  # [bs, njoints, nfeats, nframes]
            
            # target = target.expand(num_step, -1, -1, -1)
            # mask = mask.expand(num_step, -1, -1, -1)
            terms["rot_mse"] = self.masked_l2(target, sample, mask) # mean_flat(rot_mse)
            
            # terms["rot_t2m_regulization"] = self.masked_l2(target_t2m, model_output, mask_t2m)
            # target_xyz, model_output_xyz = None, None

            if weakly_style_pair:
                features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
                mu_norm = mu / mu.norm(dim=-1, keepdim=True)

                cos = cosine_sim(features_norm, mu_norm)
                cosine_loss = (1 - cos).mean()
                terms["text_cosine"] = cosine_loss

            if weakly_style_pair:     
                terms["loss"] = terms["rot_mse"].mean() + terms["text_cosine"] * 10
            else:
                # 之前 rot_t2m_regulization并未起作用 weight = 0, 一直假设0.05
                terms["loss"] = terms["rot_mse"].mean()

            return terms

        def forward_backward(self, batch, cond, style_batch, style_cond):
            self.mp_trainer.zero_grad()
            for i in range(0, batch.shape[0], self.microbatch):
                # Eliminates the microbatch feature
                assert i == 0
                assert self.microbatch == self.batch_size
                micro = batch
                micro_cond = cond
                micro_content = style_batch
                micro_style_cond = style_cond

                # last_batch = (i + self.microbatch) >= batch.shape[0]
                
                assert self.diffusion is not None

                last_batch = (i + self.microbatch) >= batch.shape[0]
                
                if self.style_finetune:
                    if self.args.use_ddim:
                        t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev(), range(int((self.args.diffusion_steps-self.args.skip_steps)/self.args.diffusion_steps * 20)))
                    else:
                        t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev(), range(self.args.diffusion_steps-self.args.skip_steps))
                else:
                    t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
    
           
                compute_losses = functools.partial(
                    self.ae_transfer_loss,
                    self.model,
                    micro,
                    t,
                    micro_content,
                    micro_style_cond["y"]["inpainted_motion"],
                    skip_steps=self.args.skip_steps,
                    model_kwargs = micro_style_cond,
                    model_t2m_kwargs = micro_cond,
                    weakly_style_pair = self.weakly_style_pair,
                    use_ddim=self.args.use_ddim,
                    
                )
            
                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.model.no_sync():
                        losses = compute_losses()

                if isinstance(self.schedule_sampler, LossAwareSampler):
                    self.schedule_sampler.update_with_local_losses(
                        t, losses["loss"].detach()
                    )
                
                if style_batch is None:
                    loss = (losses["loss"] * weights).mean()
                else:
                    loss = losses["loss"]
                
                if style_batch is None:
                    log_loss_dict(
                        self.diffusion, t, {k: v * weights for k, v in losses.items()}
                        )
                else:
                    log_loss_dict_style(
                        self.diffusion, t, losses
                    )

                self.mp_trainer.backward(loss)
    
class FinetuneDisLoop(TrainInpaintingLoop):
        def __init__(self, args, train_platform, model, data, diffusion=None, style_data=None):
            super().__init__(args, train_platform, model, data, diffusion, style_data)
        
        def _load_optimizer_state(self):
            pass

        def save(self):
            pass
        
        def _load_and_sync_parameters(self):
            # resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
            resume_checkpoint = find_resume_checkpoint(self.resume_checkpoint, 'model') if os.path.isdir(self.resume_checkpoint) else self.resume_checkpoint

            if resume_checkpoint:
                self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                missing_keys, unexpected_keys = self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    ), strict=False
                )
                assert len(unexpected_keys) == 0
                assert all([k.startswith("motion_enc.") or k.startswith("dis.") for k in missing_keys])
    
        
        def masked_l2(self, a, b, mask):
            # assuming a.shape == b.shape == bs, J, Jdim, seqlen
            # assuming mask.shape == bs, 1, 1, seqlen
            loss = self.l2_loss(a, b)
            loss = sum_flat(loss * mask.float())  # gives \sigma_euclidean over unmasked elements
            n_entries = a.shape[1] * a.shape[2]
            non_zero_elements = sum_flat(mask) * n_entries
            # print('mask', mask.shape)
            # print('non_zero_elements', non_zero_elements)
            # print('loss', loss)
            mse_loss_val = loss / non_zero_elements
            # print('mse_loss_val', mse_loss_val)
            return mse_loss_val    
        
        
        def forward_backward(self, batch, cond, style_batch, style_cond):
            self.mp_trainer.zero_grad()
            for i in range(0, batch.shape[0], self.microbatch):
                # Eliminates the microbatch feature
                assert i == 0
                assert self.microbatch == self.batch_size
                micro = batch
                micro_cond = cond
                micro_content = style_batch
                micro_style_cond = style_cond

                # last_batch = (i + self.microbatch) >= batch.shape[0]
                
                assert self.diffusion is not None

                last_batch = (i + self.microbatch) >= batch.shape[0]
                
                if self.style_finetune:
                    if self.args.use_ddim:
                        t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev(), range(int((self.args.diffusion_steps-self.args.skip_steps)/self.args.diffusion_steps * 20)))
                    else:
                        t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev(), range(self.args.diffusion_steps-self.args.skip_steps))
                else:
                    t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            
                compute_losses = functools.partial(
                    self.diffusion.ablation_dis_losses,
                    self.model,
                    micro,
                    t,
                    micro_content,
                    micro_style_cond["y"]["inpainted_motion"],
                    skip_steps=self.args.skip_steps,
                    model_kwargs = micro_style_cond,
                    model_t2m_kwargs = micro_cond,
                    weakly_style_pair = self.weakly_style_pair,
                    use_ddim=self.args.use_ddim,
                    
                )
            
                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.model.no_sync():
                        losses = compute_losses()

                if isinstance(self.schedule_sampler, LossAwareSampler):
                    self.schedule_sampler.update_with_local_losses(
                        t, losses["loss"].detach()
                    )
                
                if style_batch is None:
                    loss = (losses["loss"] * weights).mean()
                else:
                    loss = losses["loss"]
                
                if style_batch is None:
                    log_loss_dict(
                        self.diffusion, t, {k: v * weights for k, v in losses.items()}
                        )
                else:
                    log_loss_dict_style(
                        self.diffusion, t, losses
                    )

                self.mp_trainer.backward(loss)
        


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0



def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint(save_dir, mode='model'):
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    files = [file for file in os.listdir(save_dir) if (file.endswith('.pt') and file.startswith(mode))]
    steps = [int(file[len(mode):len(mode)+9]) for file in files]
    max_step = sorted(steps)[-1]
    latest_model = f"{mode}{max_step:09d}.pt"
    
    return os.path.join(save_dir, latest_model)


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

def log_loss_dict_style(diffusion, ts, losses):
    for key, values in losses.items():
        if key != 'loss':
            logger.logkv_mean(key, values.mean().item())
        else:
            logger.logkv_mean(key, values.item())

        # Log the quantiles (four quartiles, in particular).
        # for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
        #     quartile = int(4 * sub_t / diffusion.num_timesteps)
        #     logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


def log_motion_encoder_loss_dict(losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.item())

def log_motion_encoder_finetune_loss_dict(losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())