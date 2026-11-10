# Copyright (c) 2026 LongCat dNaViT Team.
# SPDX-License-Identifier: Apache-2.0

"""Training entry point for the dNaViT visual tokenizer.

Usage:
    accelerate launch --num_processes 8 scripts/train_tok.py \
        config=configs/dnavit_stageA_anyres_32x.yaml

Any config field can be overridden on the command line as a dotlist, e.g.
``dataset.params.jsonl_paths=[example_data/sample.jsonl]``.
"""
import sys
from pathlib import Path

# Add the repository root to sys.path so `utils` / `modeling` import cleanly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import os
import gc

from accelerate.utils import set_seed
from accelerate.utils import DistributedDataParallelKwargs
from accelerate import Accelerator

import torch
from omegaconf import OmegaConf
from utils.logger import setup_logger

from utils.train_utils import (
    get_config, create_model_and_loss_module,
    create_optimizer, create_lr_scheduler, create_dataloader,
    create_evaluator, auto_resume, save_checkpoint,
    init_from_pretrained_weights,
    train_one_epoch)


def main():
    workspace = os.environ.get('WORKSPACE', '')
    if workspace:
        torch.hub.set_dir(workspace + "/models/hub")

    config = get_config()

    # All run artifacts (checkpoints, logs, config.yaml, wandb) live under
    # ``results_dir/output_dir``. ``experiment.results_dir`` defaults to the
    # repository-local ``runs`` directory; an absolute ``output_dir`` (or one
    # already under results_dir) is left untouched. Mutating
    # config.experiment.output_dir here means auto_resume / save_checkpoint
    # (which read it) pick up the same path.
    results_dir = config.experiment.get("results_dir", "runs")
    _od = config.experiment.output_dir
    if not os.path.isabs(_od) and os.path.normpath(_od).split(os.sep)[0] != results_dir:
        _od = os.path.join(results_dir, _od)
    config.experiment.output_dir = _od
    output_dir = config.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)
    # Keep wandb's run files next to the checkpoints (under results_dir), not cwd.
    os.environ.setdefault("WANDB_DIR", output_dir)
    config.experiment.logging_dir = os.path.join(output_dir, "logs")

    # Whether logging to Wandb or Tensorboard.
    tracker = "tensorboard"
    if config.training.enable_wandb:
        tracker = "wandb"

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with=tracker,
        project_dir=config.experiment.logging_dir,
        split_batches=False,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )

    logger = setup_logger(
        name="Tok",
        log_level="INFO",
        output_file=f"{output_dir}/log.txt" if accelerator.is_main_process else None
    )

    # Enable TF32 on Ampere GPUs for faster training
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 enabled for faster training on Ampere GPUs")

    # Disable cudnn benchmarking for reproducibility
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed, device_specific=True)

    model, ema_model, loss_module = create_model_and_loss_module(
        config, logger, accelerator)

    # Optional warm-start from an external checkpoint (e.g. the previous stage)
    # BEFORE accelerator.prepare / auto_resume. No-op unless init_weight_path is
    # set and there is no resumable checkpoint-latest.
    init_from_pretrained_weights(config, model, ema_model, logger)

    optimizer, discriminator_optimizer = create_optimizer(config, logger, model, loss_module)

    lr_scheduler, discriminator_lr_scheduler = create_lr_scheduler(
        config, logger, accelerator, optimizer, discriminator_optimizer)

    # Fixed-res curriculum (t2i only): phase-1 renders every sample at a single
    # resolution for the first `until_step` steps, then the dataloader is rebuilt
    # for anyres. When enabled we DEFER building train_dataloader until after
    # auto_resume, so the phase is chosen from the resumed global_step (resuming
    # past the boundary goes straight to anyres and never re-enters fixed-res).
    _curri = (config.dataset.params.get("fixres_curriculum", {})
              if config.dataset.get("type", "") == "t2i" else {})
    _curri_enable = bool(_curri.get("enable", False))
    if _curri_enable:
        train_dataloader = None
    else:
        train_dataloader = create_dataloader(config, logger, accelerator)

    # Set up the reconstruction evaluator only when recon_eval will use it:
    # it builds InceptionV3 / LPIPS eagerly (and the former downloads FID
    # weights), which is wasted work -- and needs network -- otherwise. It is
    # also skipped in feature_recon_only mode (no image decoder at all).
    need_evaluator = (
        not config.model.vq_model.get("feature_recon_only", False)
        and bool(config.get("recon_eval", {}).get("enable", False))
    )
    if need_evaluator:
        evaluator = create_evaluator(config, logger, accelerator)
    else:
        evaluator = None

    # Prepare everything with accelerator
    logger.info("Preparing model, optimizer and dataloaders")
    # The dataloader is already aware of distributed training, so we don't need to prepare it.
    if discriminator_optimizer is not None:
        model, loss_module, optimizer, discriminator_optimizer, lr_scheduler, discriminator_lr_scheduler = accelerator.prepare(
            model, loss_module, optimizer, discriminator_optimizer, lr_scheduler, discriminator_lr_scheduler
        )
    else:
        # Discriminator is disabled, don't prepare discriminator components
        model, loss_module, optimizer, lr_scheduler = accelerator.prepare(
            model, loss_module, optimizer, lr_scheduler
        )
        discriminator_optimizer = None
        discriminator_lr_scheduler = None

    # Compile model after accelerator.prepare()
    if config.training.get("compile_model", False):
        logger.info("Compiling model with torch.compile")
        model = torch.compile(model)
        loss_module = torch.compile(loss_module)


    if config.training.use_ema:
        ema_model.to(accelerator.device)

    total_batch_size_without_accum = config.training.per_gpu_batch_size * accelerator.num_processes
    num_batches = math.ceil(
        config.experiment.max_train_examples / total_batch_size_without_accum)
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(num_batches / config.training.gradient_accumulation_steps)

    # Afterwards we recalculate our number of training epochs.
    # Note: We are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    # Multiply by 2 because we alternate between autoencoder and discriminator training steps
    # Each "epoch" will process the data twice: once for the autoencoder, once for the discriminator
    num_train_epochs = num_train_epochs * 2
    # Start training.
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Num epochs (for bookkeeping) = {num_train_epochs}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(f"  Instantaneous batch size per gpu = {config.training.per_gpu_batch_size}")
    total_batch_size = (
        config.training.per_gpu_batch_size *
        accelerator.num_processes *
        config.training.gradient_accumulation_steps
    )
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    global_step = 0
    first_epoch = 0

    global_step, first_epoch, wandb_run_id = auto_resume(
        config, logger, accelerator, ema_model, num_update_steps_per_epoch,
        strict= config.experiment.get("strict_loading", True))

    # Initialize trackers after auto_resume to support wandb resume
    if accelerator.is_main_process:
        init_kwargs = {
            "wandb": {
                "name": config.experiment.name
            }
        }
        # Resume wandb run if we have a run ID
        if wandb_run_id is not None:
            init_kwargs["wandb"]["id"] = wandb_run_id
            init_kwargs["wandb"]["resume"] = "allow"

        accelerator.init_trackers(
            project_name=config.experiment.project,
            init_kwargs=init_kwargs)

        # Get the wandb run ID for saving in checkpoints
        if config.training.enable_wandb:
            wandb_tracker = accelerator.get_tracker("wandb")
            wandb_run_id = wandb_tracker.run.id
            logger.info(f"Wandb run ID: {wandb_run_id}")

        config_path = Path(output_dir) / "config.yaml"
        logger.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)
        logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # Train Loop
    current_epoch = first_epoch

    # Curriculum phase resolver. Disabled => single anyres phase (force_fix_res=-1,
    # no boundary) for the whole run, identical to the pre-curriculum behavior.
    fixres_until = int(_curri.get("until_step", 0)) if _curri_enable else 0
    fixres_size = int(_curri.get("size", 256)) if _curri_enable else -1

    def _phase_for(step):
        """(force_fix_res, stop_at_step) for the phase containing `step`."""
        if _curri_enable and step < fixres_until:
            return fixres_size, fixres_until     # fixed-res warmup, stop at boundary
        return -1, None                          # anyres, run to max_train_steps

    # What the live train_dataloader was built with (-1 == anyres). For the
    # non-curriculum path the loader was already built as anyres, so seed the
    # tracker to -1 to avoid a redundant rebuild on the first iteration.
    cur_force_res = -1 if train_dataloader is not None else None

    while global_step < config.training.max_train_steps:
        force_res, stop_at = _phase_for(global_step)
        if train_dataloader is None or force_res != cur_force_res:
            # (Re)build the dataloader for the current phase. Drop the old one
            # first so its persistent workers are torn down before new workers
            # spawn (avoids leaking the 16 per-rank worker processes).
            if train_dataloader is not None:
                del train_dataloader
                gc.collect()
            train_dataloader = create_dataloader(
                config, logger, accelerator, force_fix_res=force_res)
            cur_force_res = force_res
            accelerator.print(
                f"[curriculum] dataloader built: force_fix_res={force_res}, "
                f"stop_at_step={stop_at}, global_step={global_step}")

        accelerator.print(f"Epoch {current_epoch} started (global_step={global_step}, ~{num_train_epochs} epochs estimated).")
        _step_before = global_step
        global_step = train_one_epoch(config, logger, accelerator,
                            model, ema_model, loss_module,
                            optimizer, discriminator_optimizer,
                            lr_scheduler, discriminator_lr_scheduler,
                            train_dataloader,
                            evaluator,
                            global_step,
                            wandb_run_id=wandb_run_id,
                            stop_at_step=stop_at)
        current_epoch += 1
        if global_step == _step_before:
            raise RuntimeError(
                "The training loop produced no optimizer steps: the dataloader "
                "yielded no batch. This usually means the dataset is smaller "
                "than one token budget (training.max_batch_tokens) or every "
                "sample failed to load. Check dataset.params.jsonl_paths / "
                "training.max_batch_tokens.")
        # Stop training if max steps is reached.
        if global_step >= config.training.max_train_steps:
            accelerator.print(
                f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
            )
            break

    accelerator.wait_for_everyone()
    # Save checkpoint at the end of training.
    save_checkpoint(model, output_dir, accelerator, global_step, logger=logger, wandb_run_id=wandb_run_id)
    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        if config.training.use_ema:
            ema_model.copy_to(model.parameters())
        model.save_pretrained_weight(output_dir)
    accelerator.end_training()


if __name__ == "__main__":
    main()
