"""Run configuration.

Every run is described by exactly one YAML file, and a copy of the fully resolved
configuration is written into the run directory next to the results.

The defaults below are the *code-exact* Simple Stories condition: the settings recorded
in the official repository's own run directories for the runs that produced the paper's
Table 1. Where the paper's text says something different, the comment says so, and
DEVIATIONS.md has the full story. The paper-exact condition is the same file with a
handful of fields overridden (see configs/).
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# The four auxiliary topics: the first four of the 48 Simple Stories topics in
# alphabetical order (paper Section 4 / Appendix A). Topic names are normalised the way
# the official data preparation does it: lower case, spaces replaced by dashes.
AUX_TOPICS = (
    "a-deadline-or-time-limit",
    "alien-encounters",
    "bygone-eras",
    "cultural-traditions",
)

# Names with a fixed meaning throughout the project.
CORE = "core"            # the core parameter group, the core MLP module, and the label "core"
UNLABELED = "unlabeled"  # the label given to a story that carries no label

METHODS = ("baseline", "filtering", "gram")
NOISE_KINDS = ("none", "false_negative", "false_positive", "wrong_module", "missed_unlabeled")
BATCH_ASSEMBLIES = ("single_topic", "mixed")
PRECISIONS = ("bf16", "fp32")


@dataclass
class DataConfig:
    # Directory written by scripts/prepare_data.py.
    store_dir: str = "data/simplestories"
    aux_topics: tuple[str, ...] = AUX_TOPICS
    # Fraction of the training stories, and of the validation stories, to use (the first
    # stories of every topic). Both are 1.0 for real runs; smoke tests shrink them.
    train_fraction: float = 1.0
    val_fraction: float = 1.0
    # The official code gives every topic a token budget equal to the *mean* topic size
    # of its class (the 44 core topics, or the 4 auxiliary ones), and drops what a topic
    # has beyond that. It removes just under 1% of the training data and makes one epoch
    # 16,544 steps. The paper says training uses "all available data" (DEVIATIONS 1.8).
    cap_topics_at_class_mean: bool = True
    # How sequences are grouped into batches inside one label pool:
    #   "single_topic": every batch holds sequences of one true topic. This is what the
    #                   official code does (one loader per topic file).
    #   "mixed":        sequences are shuffled across the whole pool, so the trainer uses
    #                   nothing but the given label.
    batch_assembly: str = "single_topic"


@dataclass
class LabelConfig:
    """Which label each training story is *given*. See gram/data.py: assign_labels."""

    # Fraction of training stories that carry a label, stratified by topic. The rest are
    # "unlabeled" (Stage 3).
    labeled_fraction: float = 1.0
    # Wrong labels (Stage 4), one kind at a time:
    #   false_negative   : `noise_rate` of auxiliary stories are given the label "core"
    #   false_positive   : `noise_rate` of core stories are given a random auxiliary label
    #   wrong_module     : `noise_rate` of auxiliary stories are given a different auxiliary label
    #   missed_unlabeled : the same stories as false_negative, but given no label at all
    noise: str = "none"
    noise_rate: float = 0.0


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    context_length: int = 256
    n_layers: int = 8
    n_heads: int = 8
    # Grouped-query attention: 8 query heads share 2 key/value heads. Not mentioned in the
    # paper; taken from the official code and needed to match its 26,257,920 parameters.
    n_kv_heads: int = 2
    d_model: int = 512
    eos_token_id: int = 1
    # MLP hidden width of the dense models (baseline and filtering).
    dense_mlp_dim: int = 2048
    # GRAM: one core MLP plus one auxiliary MLP per auxiliary topic in every block.
    # The paper states core 1856; the runs behind its Table 1 used 2048 (DEVIATIONS 1.1).
    core_mlp_dim: int = 2048
    aux_mlp_dim: int = 192


@dataclass
class RoutingConfig:
    """GRAM routing probabilities (paper Section 3)."""

    # Core robustness: fraction of core batches on which one auxiliary module is also
    # active (and updated).
    p_cr: float = 0.5
    # Auxiliary spread: fraction of auxiliary batches on which the core group is updated
    # as well as the batch's own module.
    p_as: float = 0.3
    # Auxiliary factor: how often each auxiliary dataset is sampled relative to once per
    # epoch. 1.0 means no over-sampling.
    p_af: float = 1.0
    # Appendix I: rescale GRAM's training length so that its total training FLOPs match
    # the baseline's. The paper's Appendix A instead says "same number of training
    # steps"; the official code does the rescaling (DEVIATIONS 1.2).
    equal_compute: bool = True


@dataclass
class OptimConfig:
    lr: float = 5e-3
    betas: tuple[float, float] = (0.9, 0.95)
    # The paper says 0.1. The official code never sets it, so PyTorch's default of 0.01
    # applied to the Table 1 runs (DEVIATIONS 1.3).
    weight_decay: float = 0.01
    warmup_fraction: float = 0.1
    decay_fraction: float = 0.1
    grad_clip: float = 1.0
    # One optimizer step uses batch_size sequences, processed as batch_size /
    # micro_batch_size micro-batches whose gradients are accumulated. Each micro-batch
    # has its own routing decision (Appendix H, "heterogeneous accumulation").
    batch_size: int = 128
    micro_batch_size: int = 128
    # "bf16": parameters, gradients and optimizer state are all stored in bfloat16, as
    # in the official code (it casts the whole model, it does not use autocast).
    # "fp32": ordinary float32 training; used by the CPU tests.
    precision: str = "bf16"
    compile: bool = False

    @property
    def accumulation_steps(self) -> int:
        return self.batch_size // self.micro_batch_size


@dataclass
class EvalConfig:
    # Forward-only batch size. It changes speed, not results.
    batch_size: int = 256
    # Number of evenly spaced validation sweeps during training. Only the baseline needs
    # these: they are the learning curves behind the compute-ratio metric.
    curve_points: int = 0
    # Each sweep scores the whole validation set of every auxiliary topic, and this many
    # training-sized batches sampled afresh from the (much larger) core validation set.
    curve_core_batches: int = 50


@dataclass
class ElicitProtocol:
    name: str = "code"
    # Number of training sequences of the forget topic the adversary finetunes on.
    num_sequences: int = 512
    # Each step is one full pass over those sequences (full-batch gradient descent).
    max_steps: int = 200
    # Stop after this many consecutive steps without a new best validation loss.
    # 0 disables early stopping and always runs max_steps.
    patience: int = 10


@dataclass
class ElicitConfig:
    enabled: bool = True
    # Finetuning learning rate = optim.lr * lr_factor. The paper does not give the
    # Simple Stories value; the official code uses stage.lr / 4 (DEVIATIONS 1.4).
    lr_factor: float = 0.25
    protocols: list[ElicitProtocol] = field(
        default_factory=lambda: [
            # What the official code runs, and what produced Table 1.
            ElicitProtocol(name="code", num_sequences=512, max_steps=200, patience=10),
            # Appendix A: "finetune each model for 75 steps using 128 sequences".
            ElicitProtocol(name="paper", num_sequences=128, max_steps=75, patience=0),
        ]
    )


@dataclass
class CheckpointConfig:
    # Checkpoints kept for later analysis: this many evenly spaced through training
    # (at 0%, 10%, ..., 90% for the default of 10), plus the final one.
    num_spaced: int = 10
    # A rolling "latest" checkpoint for surviving preemption, overwritten this often.
    resume_every_steps: int = 500


@dataclass
class ProbeConfig:
    enabled: bool = True
    # Gradient probes are taken at this many evenly spaced points (5%, 10%, ..., 100%).
    num_points: int = 20
    # A probe batch is pushed through the model this many sequences at a time, and the
    # pieces' gradients are added up. The result is the gradient of the whole batch; the
    # only effect is on memory, which matters because probes run in float32.
    chunk_size: int = 32


@dataclass
class RunConfig:
    name: str = "unnamed"
    method: str = "gram"
    # One seed fixes everything random about a run: the label split, the order of the
    # data, the routing plan and the weight initialisation. The same seed gives every
    # method the identical labeled/unlabeled split.
    seed: int = 1
    output_dir: str = "runs"
    device: str = "auto"

    data: DataConfig = field(default_factory=DataConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    elicit: ElicitConfig = field(default_factory=ElicitConfig)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)
    probes: ProbeConfig = field(default_factory=ProbeConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.name

    def validate(self) -> None:
        _require(self.method in METHODS, f"method must be one of {METHODS}, got {self.method!r}")
        _require(self.labels.noise in NOISE_KINDS, f"labels.noise must be one of {NOISE_KINDS}")
        _require(self.data.batch_assembly in BATCH_ASSEMBLIES, f"data.batch_assembly must be one of {BATCH_ASSEMBLIES}")
        _require(self.optim.precision in PRECISIONS, f"optim.precision must be one of {PRECISIONS}")
        _require(0.0 < self.data.train_fraction <= 1.0, "data.train_fraction must be in (0, 1]")
        _require(0.0 < self.data.val_fraction <= 1.0, "data.val_fraction must be in (0, 1]")
        _require(
            self.labels.noise == "none" or self.labels.labeled_fraction == 1.0,
            "wrong labels (labels.noise) are only defined with labels.labeled_fraction = 1.0",
        )
        _require(0.0 <= self.labels.labeled_fraction <= 1.0, "labels.labeled_fraction must be in [0, 1]")
        _require(0.0 <= self.labels.noise_rate <= 1.0, "labels.noise_rate must be in [0, 1]")
        _require(
            (self.labels.noise == "none") == (self.labels.noise_rate == 0.0),
            "labels.noise and labels.noise_rate must be set together",
        )
        for name in ("p_cr", "p_as"):
            _require(0.0 <= getattr(self.routing, name) <= 1.0, f"routing.{name} must be in [0, 1]")
        _require(self.routing.p_af > 0.0, "routing.p_af must be positive")
        _require(
            self.optim.batch_size % self.optim.micro_batch_size == 0,
            "optim.batch_size must be a multiple of optim.micro_batch_size",
        )
        _require(self.model.d_model % self.model.n_heads == 0, "d_model must be divisible by n_heads")
        _require(self.model.n_heads % self.model.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads")
        _require((self.model.d_model // self.model.n_heads) % 4 == 0, "head dimension must be divisible by 4")
        _require(len(set(self.data.aux_topics)) == len(self.data.aux_topics), "data.aux_topics has duplicates")
        names = [protocol.name for protocol in self.elicit.protocols]
        _require(len(set(names)) == len(names), "elicit.protocols have duplicate names")
        _require(all(name.isalnum() for name in names), "elicit protocol names must be alphanumeric")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Invalid config: {message}")


# --------------------------------------------------------------------------- #
# YAML loading and saving
# --------------------------------------------------------------------------- #


def _build(cls, data: dict, path: str):
    """Build dataclass `cls` from a plain dict, refusing keys the dataclass does not have.

    Unknown keys are an error on purpose: a misspelt option would otherwise be ignored
    and the run would silently use the default.
    """
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config: {path or 'top level'} must be a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    unknown = sorted(set(data) - set(hints))
    if unknown:
        raise ValueError(f"Invalid config: unknown key(s) {unknown} under {path or 'top level'}")

    kwargs = {}
    for key, value in data.items():
        kwargs[key] = _convert(hints[key], value, f"{path}.{key}" if path else key)
    return cls(**kwargs)


def _convert(hint, value, path: str):
    origin = typing.get_origin(hint)
    if dataclasses.is_dataclass(hint):
        return _build(hint, value, path)
    if origin is list:
        (item_hint,) = typing.get_args(hint)
        return [_convert(item_hint, item, f"{path}[{index}]") for index, item in enumerate(value)]
    if origin is tuple:
        return tuple(value)
    if hint is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def config_from_dict(data: dict) -> RunConfig:
    config = _build(RunConfig, data, "")
    config.validate()
    return config


def config_to_dict(config: RunConfig) -> dict:
    return _plain(dataclasses.asdict(config))


def _plain(value):
    """Turn tuples into lists so the YAML stays free of Python-specific tags."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def load_config(path: str | Path) -> RunConfig:
    with open(path) as handle:
        return config_from_dict(yaml.safe_load(handle))


def save_config(config: RunConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        yaml.safe_dump(config_to_dict(config), handle, sort_keys=False)
