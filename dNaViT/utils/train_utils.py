# Copyright (c) 2026 LongCat dNaViT Team.
# SPDX-License-Identifier: Apache-2.0

"""Training utils."""
import json
import os
import time
import math
from pathlib import Path
import functools
from collections import defaultdict

from data import (
    C2IImageNet, T2IJsonlDataset, t2i_collate, TokenBudgetBatchSampler,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from torch.optim import AdamW
from utils.lr_schedulers import get_scheduler
from modeling.modules import EMAModel, ReconstructionLoss
from modeling.tokenizer import (FSQTokenizer, RVQTokenizer, VQTokenizer,
                               AlignedVQTokenizer, AETokenizer, IBQTokenizer)
from evaluator import ReconstructionEvaluator

from utils.viz_utils import make_viz_from_samples


def get_config():
    """Reads configs from a yaml file and terminal."""
    cli_conf = OmegaConf.from_cli()

    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf


class AverageMeter(object):
    """Computes and stores the average and current value.
    
    This class is borrowed from
    https://github.com/pytorch/examples/blob/main/imagenet/main.py#L423
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


_TOKENIZER_REGISTRY = {
    "FSQ": FSQTokenizer,
    "RVQ": RVQTokenizer,
    "pq": RVQTokenizer,
    "PQ": RVQTokenizer,
    "VQ": VQTokenizer,
    "IBQ": IBQTokenizer,
    "AlignedVQ": AlignedVQTokenizer,
    "AE": AETokenizer,
}


def build_tokenizer(config):
    """Instantiate a tokenizer from ``config.model.vq_model.type``.

    Defaults to ``"FSQ"`` when unspecified (back-compat).
    """
    vq_type = config.model.vq_model.get("type", "FSQ")
    if vq_type not in _TOKENIZER_REGISTRY:
        raise ValueError(
            f"Unknown tokenizer type '{vq_type}'. "
            f"Supported: {list(_TOKENIZER_REGISTRY)}."
        )
    return _TOKENIZER_REGISTRY[vq_type](config)


def init_from_pretrained_weights(config, model, ema_model, logger):
    """Warm-start a fresh run from an external checkpoint (e.g. a stage-1
    feature-recon tokenizer) BEFORE accelerator.prepare / auto_resume.

    Loads ``config.experiment.init_weight_path`` into ``model`` (and the EMA
    shadow) with ``strict=False`` so newly-added modules (ResMod, the pixel
    UnifiedDecoder, the pre-quant sem projector, ...) keep their fresh init while
    the shared quantizer / codebook / FeatureDecoder weights are inherited.

    No-op unless ``init_weight_path`` is set AND there is no resumable
    ``output_dir/checkpoint-latest`` (a mid-run resume must win over warm-start).
    Operates on the RAW (un-prepared) model/ema_model.
    """
    init_path = config.experiment.get("init_weight_path", None)
    if not init_path:
        return
    # A resumable checkpoint takes precedence: don't clobber a mid-run state.
    latest = Path(config.experiment.output_dir) / "checkpoint-latest"
    if latest.exists():
        logger.info(
            f"init_weight_path set but {latest} exists; skipping warm-start "
            "(auto_resume will load the latest checkpoint).")
        return

    if os.path.isfile(init_path):
        model_file = init_path
    elif os.path.isdir(init_path):
        model_file = os.path.join(init_path, "pytorch_model.bin")
        if not os.path.isfile(model_file):
            raise ValueError(f"init_weight_path dir has no pytorch_model.bin: {init_path}")
    else:
        raise ValueError(f"init_weight_path does not exist: {init_path}")

    checkpoint = torch.load(model_file, map_location="cpu")
    cleaned = {k.replace("_orig_mod.module.", ""): v for k, v in checkpoint.items()}
    # Optional: drop keys whose name starts with one of ``init_exclude_prefixes``
    # so those submodules keep their fresh init instead of inheriting the
    # warm-start weights (e.g. stage-B re-initializes quantize.in_proj/out_proj
    # rather than reusing stage-A's continuous-AE projections).
    exclude_prefixes = tuple(config.experiment.get("init_exclude_prefixes", []))
    if exclude_prefixes:
        before = len(cleaned)
        cleaned = {k: v for k, v in cleaned.items() if not k.startswith(exclude_prefixes)}
        logger.info(f"[warm-start] excluded {before - len(cleaned)} keys matching "
                    f"prefixes {exclude_prefixes} (kept fresh-init).")
    msg = model.load_state_dict(cleaned, strict=False)
    logger.info(f"[warm-start] loaded {model_file}; missing={len(msg.missing_keys)}, "
                f"unexpected={len(msg.unexpected_keys)}")
    logger.info(f"[warm-start] missing keys (fresh-init): {msg.missing_keys}")
    if msg.unexpected_keys:
        logger.warning(f"[warm-start] unexpected keys (ignored): {msg.unexpected_keys}")

    # Optional overlay: load a SECOND checkpoint but keep only keys whose name
    # starts with one of ``init_overlay_prefixes`` (default: the pixel decoder +
    # post-quant unmerger). Used in stage-C: the base ``init_weight_path`` is the
    # stage-B feature-recon checkpoint (codebooks / projections / ResMod /
    # FeatureDecoder), and the overlay supplies the pixel ``decoder.*`` +
    # ``post_quant_unmerger.*`` that stage-B never built, taken from stage-A.
    overlay_path = config.experiment.get("init_overlay_path", None)
    if overlay_path:
        overlay_prefixes = tuple(config.experiment.get(
            "init_overlay_prefixes", ["decoder.", "post_quant_unmerger."]))
        if os.path.isfile(overlay_path):
            overlay_file = overlay_path
        elif os.path.isdir(overlay_path):
            overlay_file = os.path.join(overlay_path, "pytorch_model.bin")
            if not os.path.isfile(overlay_file):
                raise ValueError(f"init_overlay_path dir has no pytorch_model.bin: {overlay_path}")
        else:
            raise ValueError(f"init_overlay_path does not exist: {overlay_path}")
        overlay_ckpt = torch.load(overlay_file, map_location="cpu")
        overlay_clean = {k.replace("_orig_mod.module.", ""): v for k, v in overlay_ckpt.items()}
        overlay_filtered = {
            k: v for k, v in overlay_clean.items() if k.startswith(overlay_prefixes)}
        ov_msg = model.load_state_dict(overlay_filtered, strict=False)
        logger.info(
            f"[warm-start] overlay {overlay_file}; prefixes={overlay_prefixes}; "
            f"applied={len(overlay_filtered)} keys, unexpected={len(ov_msg.unexpected_keys)}")
        if ov_msg.unexpected_keys:
            logger.warning(f"[warm-start] overlay unexpected keys (ignored): {ov_msg.unexpected_keys}")

    # Frozen teacher ResMod (stage-C joint-ft): the warm-start checkpoint only
    # contains ``encoder.res_mod.*`` (the teacher twin was just fresh-built), so
    # copy the freshly-loaded student weights into the teacher and re-freeze it.
    # Done AFTER the main load / overlay but BEFORE the EMA reseed so the EMA
    # shadow captures the teacher weights too.
    enc = getattr(model, "encoder", None)
    teacher = getattr(enc, "teacher_res_mod", None) if enc is not None else None
    student = getattr(enc, "res_mod", None) if enc is not None else None
    if teacher is not None and student is not None:
        teacher.load_state_dict(student.state_dict())
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        logger.info("[warm-start] copied encoder.res_mod -> encoder.teacher_res_mod "
                    "(frozen distill teacher).")

    # Symmetric teacher-distill (16x): the FROZEN teacher LayerNorm twin mirrors
    # the frozen teacher ResMod. The checkpoint only carries the student
    # ``encoder.qwen_post_layernorm.*`` (the teacher LN was just fresh-built), so
    # copy the freshly-loaded student LN into the teacher LN and re-freeze it,
    # so the distill target LN_t(resmod_t(x)) starts identical to the student
    # normalized space. Same ordering: after load, before EMA reseed.
    teacher_ln = getattr(enc, "teacher_qwen_post_layernorm", None) if enc is not None else None
    student_ln = getattr(enc, "qwen_post_layernorm", None) if enc is not None else None
    if teacher_ln is not None and student_ln is not None:
        teacher_ln.load_state_dict(student_ln.state_dict())
        teacher_ln.eval()
        for p in teacher_ln.parameters():
            p.requires_grad_(False)
        logger.info("[warm-start] copied encoder.qwen_post_layernorm -> "
                    "encoder.teacher_qwen_post_layernorm (frozen distill teacher LN).")

    # Re-seed the EMA shadow from the warm-started weights so EMA and model
    # share the same starting point (EMAModel.shadow_params order mirrors
    # model.parameters() at construction time). Done ONCE, after any overlay.
    if ema_model is not None:
        ema_model.shadow_params = [
            p.clone().detach() for p in model.parameters()]
        logger.info("[warm-start] re-seeded EMA shadow from warm-started weights.")


def create_model_and_loss_module(config, logger, accelerator):
    """Creates a tokenizer model (FSQ/RVQ/PQ/VQ/IBQ/AlignedVQ/AE) and its loss."""
    logger.info("Creating model and loss module.")
    loss_cls = ReconstructionLoss
    model = build_tokenizer(config)
    model_cls = type(model)
    # Stage-B: freeze ResMod so only the RVQ quantizer + in/out projections +
    # FeatureDecoder train on top of a fixed {Qwen + ResMod} encoder. The
    # enc_freeze warmup only freezes trainable_blocks, not ResMod, so this is
    # a separate persistent switch handled here (before the optimizer is
    # built, so create_optimizer's requires_grad filter excludes it).
    #
    # SKIP when ``training.resmod_unfreeze_step > 0``: the phase-switch path
    # in train_one_epoch needs res_mod IN the optimizer so it can be
    # unfrozen later. Phase-1 freezes it post-optimizer-creation via
    # requires_grad toggle (AdamW skips frozen leaves, .grad stays None).
    _resmod_unfreeze = int(config.training.get("resmod_unfreeze_step", 0) or 0)
    if _resmod_unfreeze > 0 and config.model.vq_model.get("freeze_res_mod", False):
        logger.info(
            f"[freeze_res_mod] ignored because resmod_unfreeze_step="
            f"{_resmod_unfreeze} > 0; phase-1 will freeze res_mod post-"
            f"optimizer so phase-2 can unfreeze it without rebuilding the optimizer.")
    elif config.model.vq_model.get("freeze_res_mod", False):
        res_mod = getattr(getattr(model, "encoder", None), "res_mod", None)
        if res_mod is not None:
            for p in res_mod.parameters():
                p.requires_grad_(False)
            logger.info("[freeze_res_mod] froze encoder.res_mod parameters.")
        else:
            logger.warning("[freeze_res_mod] set but encoder.res_mod is None; no-op.")
    # Frozen teacher ResMod (stage-C joint-ft): always keep the teacher twin
    # frozen, on fresh start AND on resume (the warmup logic never touches
    # it, but be explicit so create_optimizer's requires_grad filter excludes
    # it and EMA never effectively updates it).
    teacher_res_mod = getattr(getattr(model, "encoder", None), "teacher_res_mod", None)
    if teacher_res_mod is not None:
        teacher_res_mod.eval()
        for p in teacher_res_mod.parameters():
            p.requires_grad_(False)
        logger.info("[teacher_res_mod] froze encoder.teacher_res_mod parameters.")
    # Frozen teacher LayerNorm twin (symmetric teacher-distill, 16x): same
    # rationale — keep it frozen on fresh start AND resume so the optimizer's
    # requires_grad filter excludes it and EMA never effectively updates it.
    teacher_ln = getattr(getattr(model, "encoder", None), "teacher_qwen_post_layernorm", None)
    if teacher_ln is not None:
        teacher_ln.eval()
        for p in teacher_ln.parameters():
            p.requires_grad_(False)
        logger.info("[teacher_qwen_post_layernorm] froze encoder.teacher_qwen_post_layernorm parameters.")
    print(model)

    # Create the EMA model.
    ema_model = None
    if config.training.use_ema:
        ema_model = EMAModel(model.parameters(), decay=0.999,
                            model_cls=model_cls, config=config)
        # Create custom saving and loading hooks so that `accelerator.save_state(...)` serializes in a nice format.
        def load_model_hook(models, input_dir):
            load_model = EMAModel.from_pretrained(os.path.join(input_dir, "ema_model"),
                                                  model_cls=model_cls, config=config)
            ema_model.load_state_dict(load_model.state_dict())
            ema_model.to(accelerator.device)
            del load_model

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                ema_model.save_pretrained(os.path.join(output_dir, "ema_model"))

        accelerator.register_load_state_pre_hook(load_model_hook)
        accelerator.register_save_state_pre_hook(save_model_hook)

    # Create the loss module (pixel-recon + optional GAN discriminator).
    loss_module = loss_cls(config=config)

    # Print model size
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info("=" * 80)
        logger.info(f"Model: {model_cls.__name__}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        if loss_module.discriminator is not None:
            disc_params = sum(p.numel() for p in loss_module.discriminator.parameters())
            logger.info(f"Discriminator parameters: {disc_params:,}")
        logger.info("=" * 80)

    return model, ema_model, loss_module


def create_optimizer(config, logger, model, loss_module):
    """Creates optimizer for the tokenizer and discriminator.

    Rules:
    - All norm layers, embedding layers, and bias terms have weight_decay=0
    - Encoder parameters use lr * encoder_lr_mult
    - Other parameters use base lr
    """
    logger.info("Creating optimizers.")
    optimizer_config = config.optimizer.params
    learning_rate = optimizer_config.learning_rate
    discriminator_learning_rate = optimizer_config.get("discriminator_learning_rate", 1e-4)

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer_cls = AdamW
    elif optimizer_type == "adamw_8bit":
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # Check if param should have weight decay 0: norm layers, embedding layers, bias terms
    def no_weight_decay(n, p):
        return (p.ndim < 2 or  # Bias terms and 1D params
                'norm' in n.lower() or 'ln' in n.lower() or  # Norm layers
                'bias' in n or  # Bias terms
                'embedding' in n.lower() or 'embed' in n.lower())  # Embedding layers

    named_parameters = list(model.named_parameters())

    # ResMod gets its own LR multiplier (res_mod_lr_mult, default 1.0) so it can
    # be fine-tuned more gently than the rest of the encoder. The LambdaLR
    # scheduler applies a scalar multiplier to each group's own initial lr, so
    # the per-group lr set here is preserved through the cosine schedule.
    res_mod_lr_mult = config.model.vq_model.get("res_mod_lr_mult", 1.0)

    def is_res_mod(n):
        return "res_mod" in n

    # Separate res_mod / encoder / non-encoder parameters
    res_mod_params_no_decay = [p for n, p in named_parameters
                               if is_res_mod(n) and no_weight_decay(n, p) and p.requires_grad]
    res_mod_params_with_decay = [p for n, p in named_parameters
                                 if is_res_mod(n) and not no_weight_decay(n, p) and p.requires_grad]

    encoder_params_no_decay = [p for n, p in named_parameters
                               if "encoder" in n and not is_res_mod(n) and no_weight_decay(n, p) and p.requires_grad]
    encoder_params_with_decay = [p for n, p in named_parameters
                                 if "encoder" in n and not is_res_mod(n) and not no_weight_decay(n, p) and p.requires_grad]

    other_params_no_decay = [p for n, p in named_parameters
                             if "encoder" not in n and no_weight_decay(n, p) and p.requires_grad]
    other_params_with_decay = [p for n, p in named_parameters
                               if "encoder" not in n and not no_weight_decay(n, p) and p.requires_grad]

    encoder_lr_mult = config.model.vq_model.get("encoder_lr_mult", 1.0)

    param_groups = [
        {"params": res_mod_params_no_decay, "weight_decay": 0., "lr": learning_rate * res_mod_lr_mult},
        {"params": res_mod_params_with_decay, "weight_decay": optimizer_config.weight_decay, "lr": learning_rate * res_mod_lr_mult},
        {"params": encoder_params_no_decay, "weight_decay": 0., "lr": learning_rate * encoder_lr_mult},
        {"params": encoder_params_with_decay, "weight_decay": optimizer_config.weight_decay, "lr": learning_rate * encoder_lr_mult},
        {"params": other_params_no_decay, "weight_decay": 0.},
        {"params": other_params_with_decay, "weight_decay": optimizer_config.weight_decay},
    ]

    # Log parameter group info
    logger.info(f"ResMod params (no decay): {len(res_mod_params_no_decay)}")
    logger.info(f"ResMod params (with decay): {len(res_mod_params_with_decay)}")
    logger.info(f"Encoder params (no decay): {len(encoder_params_no_decay)}")
    logger.info(f"Encoder params (with decay): {len(encoder_params_with_decay)}")
    logger.info(f"Other params (no decay): {len(other_params_no_decay)}")
    logger.info(f"Other params (with decay): {len(other_params_with_decay)}")
    logger.info(f"Encoder LR multiplier: {encoder_lr_mult}")
    logger.info(f"ResMod LR multiplier: {res_mod_lr_mult}")

    optimizer = optimizer_cls(
        param_groups,
        lr=learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
        eps=optimizer_config.get("eps", 1e-8)
    )

    # Discriminator optimizer (only create if discriminator is enabled)
    if hasattr(loss_module, 'use_discriminator') and not loss_module.use_discriminator:
        logger.info("Discriminator is disabled (discriminator_weight = 0), skipping discriminator optimizer creation.")
        discriminator_optimizer = None
    else:
        discriminator_named_parameters = list(loss_module.named_parameters())
        discriminator_params_no_decay = [p for n, p in discriminator_named_parameters
                                         if no_weight_decay(n, p) and p.requires_grad]
        discriminator_params_with_decay = [p for n, p in discriminator_named_parameters
                                           if not no_weight_decay(n, p) and p.requires_grad]

        discriminator_optimizer = optimizer_cls(
            [
                {"params": discriminator_params_no_decay, "weight_decay": 0.},
                {"params": discriminator_params_with_decay, "weight_decay": optimizer_config.weight_decay},
            ],
            lr=discriminator_learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            eps=optimizer_config.get("eps", 1e-8)
        )

    return optimizer, discriminator_optimizer


def create_lr_scheduler(config, logger, accelerator, optimizer, discriminator_optimizer=None):
    """Creates learning rate scheduler for the tokenizer and discriminator."""
    logger.info("Creating lr_schedulers.")
    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps * accelerator.num_processes,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps * accelerator.num_processes,
        base_lr=config.lr_scheduler.params.learning_rate,
        end_lr=config.lr_scheduler.params.end_lr,
    )
    if discriminator_optimizer is not None:
        discriminator_lr_scheduler = get_scheduler(
            config.lr_scheduler.scheduler,
            optimizer=discriminator_optimizer,
            num_training_steps=(config.training.max_train_steps  - config.losses.discriminator_start) * accelerator.num_processes,
            num_warmup_steps=config.lr_scheduler.params.warmup_steps * accelerator.num_processes,
            base_lr=config.lr_scheduler.params.learning_rate,
            end_lr=config.lr_scheduler.params.end_lr,
        )
    else:
        discriminator_lr_scheduler = None
    return lr_scheduler, discriminator_lr_scheduler


def create_dataloader(config, logger, accelerator, force_fix_res: int = -1):
    """Creates data loader for training.

    Dispatches on ``config.dataset.type``:
        - ``t2i``: anyres jsonl reader (data.t2i_dataset.T2IJsonlDataset).
        - ``c2i``: fixed-resolution ImageFolder (data.c2i_dataset.C2IImageNet).

    ``force_fix_res`` (>0) is the fixed-res curriculum override for the t2i path:
    every sample is rendered at force_fix_res x force_fix_res regardless of its
    jsonl marker. The training loop rebuilds the loader with force_fix_res=-1 to
    switch to anyres after the warmup phase.

    No eval loader is built: in-training monitoring is done by the
    ``linear_probe`` / ``recon_eval`` / ``psnr_eval`` hooks on ImageFolder data.
    """
    logger.info("Creating dataloaders.")
    total_batch_size_without_accum = config.training.per_gpu_batch_size * accelerator.num_processes
    total_batch_size = (
        config.training.per_gpu_batch_size * accelerator.num_processes * config.training.gradient_accumulation_steps
    )
    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params
    dataset_type = config.dataset.get("type", "t2i")

    # ---- c2i: fixed-resolution ImageFolder -----------------------------
    if dataset_type == "c2i":
        if config.training.get("max_batch_tokens", None) is not None:
            raise ValueError(
                "dataset.type=c2i uses fixed-resolution batching driven by "
                "training.per_gpu_batch_size; do not set training.max_batch_tokens.")
        c2i_train = C2IImageNet(
            root=dataset_config.root,
            crop_size=preproc_config.crop_size,
            resize_shorter_edge=preproc_config.get(
                "resize_shorter_edge", preproc_config.crop_size),
            random_crop=preproc_config.get("random_crop", True),
            random_flip=preproc_config.get("random_flip", True),
            normalize_mean=tuple(preproc_config.get("normalize_mean", (0.5, 0.5, 0.5))),
            normalize_std=tuple(preproc_config.get("normalize_std", (0.5, 0.5, 0.5))),
            train=True,
            max_samples=dataset_config.get("max_samples", None),
        )
        # Rank-shard the dataset so each process sees a disjoint slice per
        # epoch (the previous DataLoader(..., shuffle=True) without a
        # DistributedSampler had every rank iterate the full dataset).
        c2i_sampler = torch.utils.data.distributed.DistributedSampler(
            c2i_train,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=True,
            seed=config.training.get("seed", 0),
            drop_last=True,
        )
        train_dataloader = DataLoader(
            c2i_train,
            batch_size=config.training.per_gpu_batch_size,
            sampler=c2i_sampler,
            num_workers=dataset_config.num_workers_per_gpu,
            pin_memory=True,
            drop_last=True,
            persistent_workers=bool(dataset_config.num_workers_per_gpu),
        )
        train_dataloader.num_batches = math.ceil(
            config.experiment.max_train_examples / total_batch_size_without_accum)
        logger.info(
            f"c2i dataset: {len(c2i_train)} samples, "
            f"{train_dataloader.num_batches} batches/epoch (rank-sharded)")
        return train_dataloader

    # ---- t2i: anyres jsonl, token-budget batching -----------------------
    if dataset_type == "t2i":
        max_batch_tokens = config.training.get("max_batch_tokens", None)
        if max_batch_tokens is None:
            raise ValueError(
                "dataset.type=t2i requires training.max_batch_tokens; the "
                "varlen batch size is determined by token budget, not sample count.")
        if config.training.get("per_gpu_batch_size", None) is not None:
            logger.warning(
                "dataset.type=t2i ignores training.per_gpu_batch_size; using "
                f"max_batch_tokens={max_batch_tokens} instead.")
        jsonl_paths = dataset_config.jsonl_paths
        if isinstance(jsonl_paths, str):
            jsonl_paths = [jsonl_paths]
        t2i_train = T2IJsonlDataset(
            jsonl_paths=list(jsonl_paths),
            patch_size=dataset_config.get("patch_size", 16),
            spatial_merge_size=dataset_config.get("spatial_merge_size", 2),
            min_pixels=dataset_config.get("min_pixels", 56 * 56),
            max_pixels=dataset_config.get("max_pixels", 28 * 28 * 1280),
            max_tokens_per_sample=dataset_config.get("max_tokens_per_sample", 1024),
            normalize_mean=tuple(preproc_config.get("normalize_mean", (0.5, 0.5, 0.5))),
            normalize_std=tuple(preproc_config.get("normalize_std", (0.5, 0.5, 0.5))),
            random_flip=preproc_config.get("random_flip", True),
            shuffle_index=True,
            seed=config.training.get("seed", 0),
            force_fix_res=int(force_fix_res),
        )
        t2i_batch_sampler = TokenBudgetBatchSampler(
            t2i_train,
            max_batch_tokens=int(max_batch_tokens),
            shuffle=True,
            drop_last=True,
            seed=config.training.get("seed", 0),
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
        )
        train_dataloader = DataLoader(
            t2i_train,
            batch_sampler=t2i_batch_sampler,
            num_workers=dataset_config.num_workers_per_gpu,
            pin_memory=False,                       # list batch can't pin
            persistent_workers=bool(dataset_config.num_workers_per_gpu),
            collate_fn=t2i_collate,
        )
        # Approx — t2i batches per epoch is sampler-determined, not a fixed
        # divisor of the example count. We surface the sampler's __len__.
        train_dataloader.num_batches = max(1, len(t2i_batch_sampler))
        logger.info(
            f"t2i dataset: {len(t2i_train)} samples, "
            f"max_batch_tokens={max_batch_tokens}, "
            f"force_fix_res={int(force_fix_res)}, "
            f"~{train_dataloader.num_batches} batches/epoch/rank")
        return train_dataloader

    raise ValueError(
        f"Unknown dataset.type '{dataset_type}'. Supported: 't2i' (anyres jsonl), "
        f"'c2i' (fixed-res ImageFolder).")


def create_evaluator(config, logger, accelerator):
    """Creates the reconstruction evaluator (PSNR/SSIM/LPIPS/rFID/IS)."""
    logger.info("Creating evaluator.")
    evaluator = ReconstructionEvaluator(
        device=accelerator.device,
        enable_rfid=True,
        enable_inception_score=True,
        enable_psnr_score=True,
        enable_ssim_score=True,
        enable_lpips_score=True,
        enable_mse_error=True,
        enable_mae_error=True,
    )
    return evaluator


def auto_resume(config, logger, accelerator, ema_model,
                num_update_steps_per_epoch, strict=True):
    """Auto resuming the training from checkpoint-latest."""
    global_step = 0
    first_epoch = 0
    wandb_run_id = None
    # If resuming training.
    if config.experiment.resume:
        accelerator.wait_for_everyone()
        checkpoint_path = Path(config.experiment.output_dir) / "checkpoint-latest"

        if checkpoint_path.exists():
            logger.info(f"Found checkpoint at {checkpoint_path}")
            global_step, wandb_run_id = load_checkpoint(
                checkpoint_path,
                accelerator,
                logger=logger,
                strict=strict
            )
            logger.info(f"Resuming training from global_step={global_step}")
            if config.training.use_ema:
                ema_model.set_step(global_step)
            first_epoch = global_step // num_update_steps_per_epoch
            logger.info(f"Resuming from epoch {first_epoch}")
        else:
            logger.info("No checkpoint found. Training from scratch.")
    return global_step, first_epoch, wandb_run_id


# Tokenizer Training Epoch
def train_one_epoch(config, logger, accelerator,
                    model, ema_model, loss_module,
                    optimizer, discriminator_optimizer,
                    lr_scheduler, discriminator_lr_scheduler,
                    train_dataloader,
                    evaluator,
                    global_step,
                    wandb_run_id=None,
                    stop_at_step=None):

    """One epoch training.

    ``stop_at_step``: optional hard boundary (e.g. the fixed-res curriculum
    switch point). When the running ``global_step`` reaches it, the epoch
    returns early so the caller can rebuild the dataloader for the next phase.
    """
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    end = time.time()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    if discriminator_optimizer is not None:
        discriminator_optimizer.zero_grad(set_to_none=True)

    # Drive epoch counter into DistributedSampler / TokenBudgetBatchSampler so
    # each epoch reshuffles. WebDataset has its own internal shuffling and
    # doesn't expose set_epoch.
    epoch_seed = global_step  # monotonic; works as a per-epoch shuffle key
    sampler = getattr(train_dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch_seed)
    batch_sampler = getattr(train_dataloader, "batch_sampler", None)
    if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch_seed)

    autoencoder_logs = defaultdict(float)
    discriminator_logs = defaultdict(float)
    train_autoencoder_next = True

    # Feature-reconstruction-only mode: tokenizer has no image decoder, so the
    # pixel-reconstruction / evaluation steps below are skipped.
    feature_recon_only = config.model.vq_model.get("feature_recon_only", False)

    # Encoder-freeze warmup. The trainable backbone is frozen for the first
    # ``enc_freeze_warmup_step`` steps so the random decoder + freshly built
    # ResMod / quantizer / clip projectors converge without dragging the
    # pretrained encoder representation off-manifold. The optimizer was
    # built with all encoder params included (defaults to requires_grad=True
    # at construction); we just toggle requires_grad on/off so AdamW skips
    # frozen leaves (their .grad stays None).
    enc_freeze_step = int(config.training.get("enc_freeze_warmup_step", 0) or 0)
    if enc_freeze_step != 0:
        unwrapped = accelerator.unwrap_model(model)
        if enc_freeze_step < 0:
            # enc_freeze_warmup_step < 0: keep the encoder backbone FROZEN for the
            # ENTIRE run. Only ResMod / decoder / quantizer / clip projectors train
            # (ResMod is a separate module, untouched by set_encoder_trainable).
            unwrapped.set_encoder_trainable(False)
            if accelerator.is_main_process:
                logger.info(
                    "[enc_freeze_warmup] encoder backbone FROZEN for the whole run "
                    "(enc_freeze_warmup_step<0); training only ResMod/decoder/quantizer/projectors")
        elif global_step < enc_freeze_step:
            unwrapped.set_encoder_trainable(False)
            if accelerator.is_main_process:
                logger.info(
                    f"[enc_freeze_warmup] encoder backbone FROZEN until step "
                    f"{enc_freeze_step} (current global_step={global_step})")
        else:
            unwrapped.set_encoder_trainable(True)
            if accelerator.is_main_process:
                logger.info(f"encoder backbone TRAINABLE (current global_step={global_step})")

    # --- ResMod phase switch (stage-B -> stage-C merge) -------------------
    # When ``training.resmod_unfreeze_step > 0``: phase-1 freezes the student
    # ResMod and detaches the pixel decoder (stage-B behavior); at the step
    # boundary phase-2 unfreezes ResMod (student trains at res_mod_lr_mult)
    # and attaches the pixel decoder (pixel grads flow into RVQ, stage-C
    # behavior). The optimizer was built with res_mod params included
    # (requires_grad=True at construction); we toggle requires_grad so AdamW
    # skips frozen leaves in phase-1 (their .grad stays None). Re-applied on
    # every entry to train_one_epoch so resume past the boundary goes
    # straight to phase-2. 0 = no phase switch (run phase-2 behavior throughout).
    resmod_unfreeze_step = int(config.training.get("resmod_unfreeze_step", 0) or 0)
    _phase2_active = False
    if resmod_unfreeze_step > 0:
        unwrapped = accelerator.unwrap_model(model)
        res_mod = getattr(getattr(unwrapped, "encoder", None), "res_mod", None)
        if global_step < resmod_unfreeze_step:
            # Phase-1: freeze student ResMod + detach pixel decoder.
            if res_mod is not None:
                for p in res_mod.parameters():
                    p.requires_grad_(False)
            unwrapped.detach_pixel_decoder = True
            if accelerator.is_main_process:
                logger.info(
                    f"[phase] phase-1 (stage-B): res_mod frozen, "
                    f"detach_pixel_decoder=True (global_step={global_step}, "
                    f"boundary={resmod_unfreeze_step})")
        else:
            # Resume past the boundary: phase-2 already active.
            _phase2_active = True
            if accelerator.is_main_process:
                logger.info(
                    f"[phase] phase-2 (stage-C) active on entry "
                    f"(global_step={global_step} >= boundary {resmod_unfreeze_step})")

    for _, batch in enumerate(train_dataloader):
        model.train()
        # Unfreeze the encoder backbone the first iter we hit on/after the
        # warmup threshold. Cheap idempotent toggle. (enc_freeze_step < 0 stays
        # frozen for the whole run, so this never fires for it.)
        if enc_freeze_step > 0 and global_step == enc_freeze_step:
            accelerator.unwrap_model(model).set_encoder_trainable(True)
            if accelerator.is_main_process:
                logger.info(
                    f"[enc_freeze_warmup] unfreezing encoder backbone at "
                    f"global_step={global_step}")
        # ResMod phase-2 unfreeze at the boundary (stage-B -> stage-C).
        # Idempotent: only fires once when global_step hits the boundary.
        if resmod_unfreeze_step > 0 and global_step == resmod_unfreeze_step \
                and not _phase2_active:
            unwrapped = accelerator.unwrap_model(model)
            res_mod = getattr(getattr(unwrapped, "encoder", None), "res_mod", None)
            if res_mod is not None:
                for p in res_mod.parameters():
                    p.requires_grad_(True)
            unwrapped.detach_pixel_decoder = False
            _phase2_active = True
            if accelerator.is_main_process:
                logger.info(
                    f"[phase] phase-1 -> phase-2 at global_step={global_step}: "
                    f"res_mod unfrozen (lr_mult="
                    f"{config.model.vq_model.get('res_mod_lr_mult', 1.0)}), "
                    f"detach_pixel_decoder=False")
        # batch["image"] is either a dense tensor (fixed-res / c2i)
        # or a list of per-sample tensors (anyres / t2i). The varlen path in
        # the tokenizer is selected automatically when image_grid_thw is set.
        raw_image = batch["image"]
        if isinstance(raw_image, (list, tuple)):
            images = [t.to(accelerator.device, non_blocking=True) for t in raw_image]
            image_grid_thw = batch["image_grid_thw"].to(accelerator.device)
        else:
            images = raw_image.to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )
            image_grid_thw = None
        fnames = batch["__key__"]
        data_time_meter.update(time.time() - end)
        should_train_discriminator = (global_step >= config.losses.discriminator_start and
                                     accelerator.unwrap_model(loss_module).use_discriminator)
        train_autoencoder = train_autoencoder_next or not should_train_discriminator

        def _model_forward():
            if image_grid_thw is not None:
                return model(images, image_grid_thw=image_grid_thw)
            return model(images)

        if train_autoencoder:
            reconstructed_images, extra_results_dict = _model_forward()
        else:
            with torch.no_grad():
                model.eval()
                reconstructed_images, extra_results_dict = _model_forward()
                model.train()

        if train_autoencoder:
            autoencoder_logs = defaultdict(float)
            with accelerator.accumulate([model]):
                autoencoder_loss, loss_dict = loss_module(
                    images,
                    reconstructed_images,
                    extra_results_dict,
                    global_step,
                    mode="generator"
                )

                # Gather the losses across all processes for logging.
                autoencoder_logs = {}
                for k, v in loss_dict.items():
                    if k in ["discriminator_factor", "d_weight"]:
                        if type(v) == torch.Tensor:
                            autoencoder_logs["train/" + k] = v.cpu().item()
                        else:
                            autoencoder_logs["train/" + k] = v
                    else:
                        autoencoder_logs["train/" + k] = accelerator.gather(v).mean().item()

                accelerator.backward(autoencoder_loss)

                if accelerator.sync_gradients:
                    if config.training.max_grad_norm is not None:
                        accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                # Log gradient norm before zeroing it.
                if (
                    accelerator.sync_gradients
                    and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                    and accelerator.is_main_process
                ):
                    log_grad_norm(model, accelerator, global_step + 1)
        else:
            discriminator_logs = defaultdict(float)
            with accelerator.accumulate(loss_module):
                # Discriminator training.
                discriminator_loss, loss_dict_discriminator = loss_module(
                    images,
                    reconstructed_images,
                    extra_results_dict,
                    global_step=global_step,
                    mode="discriminator",
                )

                # Gather the losses across all processes for logging.
                for k, v in loss_dict_discriminator.items():
                    if k in ["logits_real", "logits_fake"]:
                        if type(v) == torch.Tensor:
                            discriminator_logs["train/" + k] = v.cpu().item()
                        else:
                            discriminator_logs["train/" + k] = v
                    else:
                        discriminator_logs["train/" + k] = accelerator.gather(v).mean().item()

                accelerator.backward(discriminator_loss)

                if accelerator.sync_gradients and discriminator_optimizer is not None:
                    if config.training.max_grad_norm is not None:
                        accelerator.clip_grad_norm_(loss_module.parameters(), config.training.max_grad_norm)
                    discriminator_optimizer.step()
                    if discriminator_lr_scheduler is not None:
                        discriminator_lr_scheduler.step()
                    discriminator_optimizer.zero_grad(set_to_none=True)
        
                # Log gradient norm before zeroing it.
                if (
                    accelerator.sync_gradients
                    and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                    and accelerator.is_main_process
                ):
                    log_grad_norm(loss_module, accelerator, global_step + 1)

        # ``not train_autoencoder`` means both an autoencoder step and a
        # discriminator step were completed in this iteration.
        if accelerator.sync_gradients:
            if should_train_discriminator:
                train_autoencoder_next = not train_autoencoder_next
            else:
                train_autoencoder_next = True

            # NOTE: ema_model.step runs on every sync_gradients tick (including
            # discriminator-only iterations) so EMA's internal
            # ``optimization_step`` stays aligned with ``global_step``, which is
            # what auto_resume relies on via ``ema_model.set_step(global_step)``.
            # When the discriminator is active this advances the EMA warmup
            # schedule slightly faster than the number of autoencoder updates;
            # tune EMA decay / warmup_steps if you train past
            # discriminator_start.
            if config.training.use_ema:
                ema_model.step(model.parameters())
            batch_time_meter.update(time.time() - end)
            end = time.time()

            if (global_step + 1) % config.experiment.log_every == 0:
                samples_per_second_per_gpu = (
                    config.training.gradient_accumulation_steps * config.training.per_gpu_batch_size / batch_time_meter.val
                )

                # Report the main (highest) training LR rather than param_groups[0].
                # group 0 can be an empty ResMod group or the encoder group (scaled
                # by encoder_lr_mult, e.g. 0.1x) depending on run vintage, which
                # misleadingly showed a 10x-smaller LR in the log. max() always
                # picks the primary quantizer/proj group (full base LR).
                lr = max(lr_scheduler.get_last_lr())
                dis_lr = max(discriminator_lr_scheduler.get_last_lr()) if discriminator_lr_scheduler is not None else 0.0

                log_msg = (
                    f"Data (t): {data_time_meter.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                    f"Batch (t): {batch_time_meter.val:0.4f} "
                    f"LR: {lr:0.6f} "
                    f"Step: {global_step + 1} "
                    f"Total: {autoencoder_logs['train/total_loss']:0.4f} "
                )

                # Detailed sub-item breakdown. Sub-totals (Recon / Quant /
                # Sem) print first with their constituents on the right
                # (e.g. "Recon(L1+L2): 0.28 [L1=0.31 L2=0.25]") so the user
                # can see both the aggregate and the contributing pieces
                # without having to dig into wandb.
                def _fmt_group(autoencoder_logs, total_key, total_lbl, components):
                    """Append "TotalLbl: x.xxx [c1=y.yyy c2=z.zzz]" if total
                    is present and non-zero."""
                    tkey = f"train/{total_key}"
                    if tkey not in autoencoder_logs:
                        return ""
                    tv = autoencoder_logs[tkey]
                    if tv == 0.0:
                        return ""
                    parts = []
                    for ck, clbl in components:
                        fkey = f"train/{ck}"
                        if fkey in autoencoder_logs:
                            cv = autoencoder_logs[fkey]
                            if cv != 0.0:
                                parts.append(f"{clbl}={cv:0.4f}")
                    if parts:
                        return f"{total_lbl}: {tv:0.4f} [{' '.join(parts)}] "
                    return f"{total_lbl}: {tv:0.4f} "

                # Reconstruction: weighted-sum total + raw L1 / L2 components.
                log_msg += _fmt_group(
                    autoencoder_logs, "reconstruction_loss", "Recon",
                    [("recon_l1", "L1"), ("recon_l2", "L2")])
                # Quantizer-internal: total + commitment + vq + entropy + cb.
                log_msg += _fmt_group(
                    autoencoder_logs, "quantizer_loss", "Quant",
                    [("commitment_loss", "Commit"),
                     ("vq_loss", "VQ"),
                     ("entropy_loss", "Entropy"),
                     ("codebook_loss", "CB")])
                # Semantic distillation: total + pre + post.
                log_msg += _fmt_group(
                    autoencoder_logs, "semantic_total", "Sem",
                    [("pre_sem_loss", "Pre"),
                     ("pre_sem_mse_loss", "PreMSE"),
                     ("post_sem_loss", "Post")])

                # Stand-alone weighted losses (no sub-components).
                _solo_keys = [
                    ("feature_recon_loss", "FeatRec"),
                    ("feature_distill_loss", "FeatDistill"),
                    ("feature_distill_cos_loss", "FeatDistillCos"),
                    ("resmod_preserve_loss", "ResModCos"),
                    ("resmod_preserve_mse_loss", "ResModMSE"),
                    ("perceptual_loss", "Percep"),
                    ("clip_loss", "Clip"),
                    ("gram_loss", "Gram"),
                ]
                for k, lbl in _solo_keys:
                    fkey = f"train/{k}"
                    if fkey in autoencoder_logs:
                        v = autoencoder_logs[fkey]
                        if v != 0.0:
                            log_msg += f"{lbl}: {v:0.4f} "

                # Codebook utilization, per-group commit / vq summary.
                # Expand each group / depth explicitly so multi-codebook
                # configs (shared_group_num > 1 or no sharing at all) don't
                # collapse to a single number. Keys come back in their
                # natural g0/g1 (or d0..d7 / shared) order from
                # _update_and_get_usage; sort for determinism.
                def _collect(prefix):
                    items = [
                        (k[len(prefix):], v)
                        for k, v in autoencoder_logs.items()
                        if k.startswith(prefix)
                    ]
                    return sorted(items, key=lambda kv: kv[0])

                _cb = _collect("train/codebook_usage/")
                if _cb:
                    log_msg += "CB-util[" + " ".join(
                        f"{lbl}={v:0.3f}" for lbl, v in _cb) + "] "
                _cg = _collect("train/commit_per_group/")
                if _cg:
                    log_msg += "Commit-by-group[" + " ".join(
                        f"{lbl}={v:0.4f}" for lbl, v in _cg) + "] "
                _vg = _collect("train/vq_per_group/")
                if _vg:
                    log_msg += "VQ-by-group[" + " ".join(
                        f"{lbl}={v:0.4f}" for lbl, v in _vg) + "] "

                if should_train_discriminator:
                    log_msg += (
                        f"Gen GAN Loss: {autoencoder_logs['train/gan_loss']:0.4f} "
                        f"Discr Loss: {discriminator_logs['train/discriminator_loss']:0.4f} "
                    )

                logger.info(log_msg
                )
                logs = {
                    "lr": lr,
                    "lr/autoencoder": lr,
                    "lr/discriminator": dis_lr,
                    "samples/sec/gpu": samples_per_second_per_gpu,
                    "time/data_time": data_time_meter.val,
                    "time/batch_time": batch_time_meter.val,
                    # Removed pass_through_ratio - FSQ doesn't have this feature
                }
                logs.update(autoencoder_logs)
                logs.update(discriminator_logs)
                accelerator.log(logs, step=global_step + 1)

                # Reset batch / data time meters per log window.
                batch_time_meter.reset()
                data_time_meter.reset()

            # Save model checkpoint.
            if (global_step + 1) % config.experiment.save_every == 0:
                save_path = save_checkpoint(
                    model, config.experiment.output_dir, accelerator, global_step + 1, logger=logger, wandb_run_id=wandb_run_id)
                # Wait for everyone to save their checkpoint.
                accelerator.wait_for_everyone()

            # Generate images. Skipped in feature_recon_only mode: there is no
            # image decoder to reconstruct pixels from.
            if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process \
                    and not feature_recon_only:
                # Store the model parameters temporarily and load the EMA parameters to perform inference.
                if config.training.get("use_ema", False):
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())

                if isinstance(images, list):
                    reconstruct_images_anyres(
                        model,
                        images[:config.training.num_generated_images],
                        image_grid_thw[:config.training.num_generated_images],
                        fnames[:config.training.num_generated_images],
                        accelerator,
                        global_step + 1,
                        config.experiment.output_dir,
                        logger=logger,
                        config=config,
                    )
                else:
                    reconstruct_images(
                        model,
                        images[:config.training.num_generated_images],
                        fnames[:config.training.num_generated_images],
                        accelerator,
                        global_step + 1,
                        config.experiment.output_dir,
                        logger=logger,
                        config=config,
                    )

                if config.training.get("use_ema", False):
                    # Switch back to the original model parameters for training.
                    ema_model.restore(model.parameters())


            # Linear-probe monitor (every eval_every): measures pre/post-quant
            # feature separability on a fixed ImageNet subset. Runs on EMA weights
            # under no_grad/eval (no codebook update), main process only, wrapped
            # in try/except — never disturbs training. Independent of the pixel
            # eval below (not gated on feature_recon_only).
            if (global_step + 1) % config.experiment.eval_every == 0 \
                    and config.get("linear_probe", {}).get("enable", False):
                # Belt-and-suspenders: run_linear_probe is itself fully
                # try/except-wrapped, but guard the import/call site too so that
                # NOTHING from the probe (a bad import, an OOM in the finally
                # block, etc.) can ever take the training loop down. Worst case we
                # log a warning and keep training — offline probing still works.
                try:
                    from utils.linear_probe import run_linear_probe
                    run_linear_probe(config, model, ema_model, accelerator, logger, global_step + 1)
                except Exception as _lp_err:  # noqa: BLE001
                    logger.warning(f"[linear_probe] call skipped due to error: {_lp_err!r}")

            # Reconstruction-metric monitor (every eval_every): the t2i/anyres
            # path builds no eval loader, so this hook is the only per-eval
            # reconstruction metric. It reports PSNR/SSIM/LPIPS/rFID on the
            # same fixed-res ImageNet subset as the linear probe, on EMA
            # weights, main process only, fully try/except-wrapped.
            if (global_step + 1) % config.experiment.eval_every == 0 \
                    and config.get("recon_eval", {}).get("enable", False):
                try:
                    from utils.recon_eval import run_recon_eval
                    run_recon_eval(config, model, ema_model, accelerator, evaluator, logger, global_step + 1)
                except Exception as _re_err:  # noqa: BLE001
                    logger.warning(f"[recon_eval] call skipped due to error: {_re_err!r}")

            # Legacy PSNR-only monitor. Kept beside recon_eval so existing
            # configurations that request the full metric suite are unchanged.
            if (global_step + 1) % config.experiment.eval_every == 0 \
                    and not feature_recon_only \
                    and config.get("psnr_eval", {}).get("enable", False):
                try:
                    from utils.psnr_eval import run_psnr_eval
                    run_psnr_eval(
                        config, model, ema_model, accelerator, logger, global_step + 1
                    )
                except Exception as _pe_err:  # noqa: BLE001
                    logger.warning(f"[psnr_eval] call skipped due to error: {_pe_err!r}")

            # Sync all ranks after the main-process-only monitors (linear_probe,
            # recon_eval, or psnr_eval) finish. Without this, non-rank0 workers
            # re-enter the training loop while rank0 is still running eval and
            # hit quantizer collectives (EMA all_reduce, dead-code restart
            # broadcast) whose rank0 counterpart is skipped because model.eval()
            # short-circuits them — leading to an NCCL hang. The condition is
            # identical on every rank (config constants), so every rank enters
            # the barrier together.
            if (global_step + 1) % config.experiment.eval_every == 0 and (
                config.get("linear_probe", {}).get("enable", False)
                or config.get("recon_eval", {}).get("enable", False)
                or config.get("psnr_eval", {}).get("enable", False)
            ):
                accelerator.wait_for_everyone()

            global_step += 1
            if global_step >= config.training.max_train_steps:
                accelerator.print(
                    f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
                )
                break
            if stop_at_step is not None and global_step >= stop_at_step:
                accelerator.print(
                    f"Reached curriculum boundary at step {global_step} "
                    f"(stop_at_step={stop_at_step}); returning to rebuild dataloader."
                )
                break

    return global_step


@torch.no_grad()
@torch._dynamo.disable
def reconstruct_images(model, original_images, fnames, accelerator,
                    global_step, output_dir, logger, config=None, text=None):
    logger.info("Reconstructing images...")
    original_images = torch.clone(original_images)
    model.eval()
    dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16

    # Use autocast for FlashAttention compatibility
    with torch.autocast("cuda", dtype=dtype, enabled=accelerator.mixed_precision != "no"):
        enc_tokens, encoder_dict = accelerator.unwrap_model(model).encode(original_images)
        reconstructed_images = accelerator.unwrap_model(model).decode(enc_tokens)[0]

    # shift both to [0, 1]
    original_images = (original_images + 1.) / 2.
    reconstructed_images = (reconstructed_images + 1.) / 2.

    images_for_saving, images_for_logging = make_viz_from_samples(
        original_images,
        reconstructed_images
    )

    # Log images.
    if config.training.enable_wandb:
        accelerator.get_tracker("wandb").log_images(
            {f"Train Reconstruction": images_for_saving},
            step=global_step
        )
    else:
        accelerator.get_tracker("tensorboard").log_images(
            {"Train Reconstruction": images_for_logging}, step=global_step
        )
    # Log locally.
    root = Path(output_dir) / "train_images"
    os.makedirs(root, exist_ok=True)
    for i,img in enumerate(images_for_saving):
        # Reduce fname to its stem so a path-like key can't inject subdirs.
        fname = Path(str(fnames[i])).stem
        filename = f"{global_step:08}_s-{i:03}-{fname}.png"
        path = os.path.join(root, filename)
        img.save(path)

    model.train()


@torch.no_grad()
def reconstruct_images_anyres(model, images_list, image_grid_thw, fnames, accelerator,
                    global_step, output_dir, logger, config=None):
    """Anyres/varlen counterpart of ``reconstruct_images``.

    ``images_list`` is a list of per-sample (3, H_i, W_i) tensors in [-1, 1]
    with differing sizes, so each image is encoded/decoded and saved
    individually (``make_viz_from_samples`` can only stack same-size batches).
    Mirrors the varlen forward (encode -> decode with image_grid_thw).
    """
    logger.info("Reconstructing images (anyres)...")
    model.eval()
    dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16

    n = min(len(images_list), config.training.num_generated_images)
    sub_imgs = [images_list[i].to(accelerator.device, non_blocking=True) for i in range(n)]
    sub_grid = image_grid_thw[:n].to(accelerator.device)

    unwrapped = accelerator.unwrap_model(model)
    with torch.autocast("cuda", dtype=dtype, enabled=accelerator.mixed_precision != "no"):
        if getattr(unwrapped, "continuous", False):
            z_quantized, _ = unwrapped.encode_continuous(sub_imgs, image_grid_thw=sub_grid)
        else:
            z_quantized, _ = unwrapped.encode(sub_imgs, image_grid_thw=sub_grid)
        recon_list = unwrapped.decode(z_quantized, image_grid_thw=sub_grid)[0]

    root = Path(output_dir) / "train_images"
    os.makedirs(root, exist_ok=True)
    pil_imgs = []
    for i in range(n):
        rec = recon_list[i]
        if rec.dim() == 4:
            rec = rec.squeeze(0)
        # shift [-1, 1] -> [0, 1]
        orig01 = (sub_imgs[i].float() + 1.) / 2.
        rec01 = (rec.float() + 1.) / 2.
        # Safety: per-image sizes should already match; guard anyway.
        if rec01.shape[-2:] != orig01.shape[-2:]:
            rec01 = F.interpolate(rec01.unsqueeze(0), size=orig01.shape[-2:],
                                  mode="bicubic", align_corners=False).squeeze(0)
        images_for_saving, _ = make_viz_from_samples(
            orig01.unsqueeze(0), rec01.unsqueeze(0))
        img = images_for_saving[0]
        fname = fnames[i] if fnames is not None and i < len(fnames) else f"img{i}"
        # ``fname`` may be a full dataset path (e.g. /data/val/000872926.jpg);
        # keep only the stem so os.path.join doesn't treat its '/' as subdirs
        # (which would point at a non-existent directory) or emit '.jpg.png'.
        fname = Path(str(fname)).stem
        img.save(os.path.join(root, f"{global_step:08}_s-{i:03}-{fname}.png"))
        pil_imgs.append(img)

    if config is not None and config.training.enable_wandb and pil_imgs:
        accelerator.get_tracker("wandb").log_images(
            {"Train Reconstruction (anyres)": pil_imgs}, step=global_step)

    model.train()


def save_checkpoint(model, output_dir, accelerator, global_step, logger, wandb_run_id=None) -> Path:
    """Save a PERMANENT per-step checkpoint and refresh ``checkpoint-latest``.

    Each call writes ``checkpoint-{global_step}`` (kept forever — one snapshot
    per ``save_every``) using a temp-dir + atomic-rename so a crash mid-save can
    never corrupt an existing checkpoint. ``checkpoint-latest`` is then re-pointed
    at it as a symlink, so ``auto_resume`` / the warm-start gate (both probe
    ``checkpoint-latest.exists()``, which follows symlinks) keep working
    unchanged.
    """
    import shutil

    output_dir = Path(output_dir)
    final_path = output_dir / f"checkpoint-{global_step}"
    temp_path = output_dir / f"checkpoint-{global_step}-new"
    latest_link = output_dir / "checkpoint-latest"

    # Save to temporary location first
    state_dict = accelerator.get_state_dict(model)
    # Clear any stale temp dir from a previously-failed save (main only), then
    # have EVERY rank create temp_path before save_state: accelerator.save_state
    # makes each rank write its own random_states_{rank}.pkl into temp_path, so
    # the dir must exist on all ranks (not just where unwrapped_model is written).
    if accelerator.is_main_process and temp_path.exists():
        shutil.rmtree(temp_path)
    accelerator.wait_for_everyone()
    os.makedirs(temp_path, exist_ok=True)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained_weight(
            temp_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
        )
        metadata = {"global_step": global_step}
        if wandb_run_id is not None:
            metadata["wandb_run_id"] = wandb_run_id
        json.dump(metadata, (temp_path / "metadata.json").open("w+"))

    accelerator.save_state(temp_path)

    # Wait for all processes to finish saving
    accelerator.wait_for_everyone()

    # Atomically promote the temp dir to its permanent name, then re-point the
    # checkpoint-latest symlink (only main process).
    if accelerator.is_main_process:
        if final_path.exists():
            # Re-running the same step (e.g. after a resume): replace it.
            logger.info(f"Removing existing checkpoint at {final_path}")
            shutil.rmtree(final_path)
        temp_path.rename(final_path)

        # Refresh checkpoint-latest -> checkpoint-{step} (relative symlink so the
        # output_dir stays relocatable). Tolerate a pre-existing real dir from
        # older runs by removing it first.
        if latest_link.is_symlink() or latest_link.exists():
            if latest_link.is_symlink() or latest_link.is_file():
                latest_link.unlink()
            else:
                shutil.rmtree(latest_link)
        os.symlink(final_path.name, latest_link)
        logger.info(
            f"Saved checkpoint to {final_path} (global_step={global_step}); "
            f"checkpoint-latest -> {final_path.name}")

    # Wait for rename to complete
    accelerator.wait_for_everyone()

    return final_path


def load_checkpoint(checkpoint_path: Path, accelerator, logger, strict=True):
    logger.info(f"Load checkpoint from {checkpoint_path}")

    accelerator.load_state(checkpoint_path, strict=strict)

    with open(checkpoint_path / "metadata.json", "r") as f:
        metadata = json.load(f)
        global_step = int(metadata["global_step"])
        wandb_run_id = metadata.get("wandb_run_id", None)

    logger.info(f"Resuming at global_step {global_step}")
    if wandb_run_id:
        logger.info(f"Resuming wandb run with ID: {wandb_run_id}")
    return global_step, wandb_run_id


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)

