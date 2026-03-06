import shutil
import uuid
import time
import logging
import yaml
from rich.logging import RichHandler
from rich.table import Table
from rich.console import Console
import rich.box
from typing import Dict, List, Optional, Union, cast
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from subprocess import Popen
from cuco.launch import JobScheduler, JobConfig, ProcessWithLogging
from cuco.database import ProgramDatabase, DatabaseConfig, Program
from cuco.llm import (
    LLMClient,
    extract_between,
    EmbeddingClient,
    BanditBase,
    AsymmetricUCB,
)
from cuco.edit import (
    apply_diff_patch,
    apply_full_patch,
    summarize_diff,
    redact_immutable,
)
from cuco.core.sampler import PromptSampler
from cuco.core.summarizer import MetaSummarizer
from cuco.core.novelty_judge import NoveltyJudge
from cuco.logo import print_gradient_logo

FOLDER_PREFIX = "gen"


@dataclass
class EvolutionConfig:
    task_sys_msg: Optional[str] = None
    patch_types: List[str] = field(default_factory=lambda: ["diff"])
    patch_type_probs: List[float] = field(default_factory=lambda: [1.0])
    num_generations: int = 10
    max_parallel_jobs: int = 2
    max_patch_resamples: int = 3
    max_patch_attempts: int = 5
    job_type: str = "local"
    language: str = "python"
    llm_models: List[str] = field(default_factory=lambda: ["azure-gpt-4.1-mini"])
    llm_dynamic_selection: Optional[Union[str, BanditBase]] = None
    llm_dynamic_selection_kwargs: dict = field(default_factory=lambda: {})
    llm_kwargs: dict = field(default_factory=lambda: {})
    meta_rec_interval: Optional[int] = None
    meta_llm_models: Optional[List[str]] = None
    meta_llm_kwargs: dict = field(default_factory=lambda: {})
    meta_max_recommendations: int = 5
    embedding_model: Optional[str] = None
    init_program_path: Optional[str] = None
    results_dir: Optional[str] = None
    max_novelty_attempts: int = 3
    code_embed_sim_threshold: float = 1.0
    novelty_llm_models: Optional[List[str]] = None
    novelty_llm_kwargs: dict = field(default_factory=lambda: {})
    use_text_feedback: bool = False

    # Host-to-device pre-transformation settings
    pre_transform_enabled: bool = False  # Auto-transform host-side NCCL to GIN before evolution
    pre_transform_max_iterations: int = 20
    pre_transform_two_stage: bool = True  # Stage A: infrastructure, Stage B: kernel replacement
    pre_transform_stage_a_max_iterations: int = 5
    pre_transform_stage_b_max_iterations: int = 10
    pre_transform_rewrite_model: Optional[str] = None  # Defaults to first llm_models entry
    pre_transform_judge_model: str = ""  # Empty = use same model as rewriter (single LLM)
    pre_transform_reference_code_path: Optional[str] = None  # Path to reference GIN code
    pre_transform_nccl_api_docs: str = ""  # NCCL API docs string for the rewriter LLM
    pre_transform_agent: bool = False  # Use Claude Code agent (full autonomy) for pre-transform
    pre_transform_agent_model: str = "opus"  # Claude model alias for agent mode
    pre_transform_agent_prompt_builder: Optional[object] = None  # Callable(source_path, work_dir) -> str
    pre_transform_agent_system_prompt: str = ""  # System prompt override for agent mode

    # Per-island customization (multi-API evolution)
    task_sys_msg_per_island: Optional[Dict[int, str]] = None
    init_program_paths_per_island: Optional[Dict[int, str]] = None
    reference_code_per_island: Optional[Dict[int, str]] = None

    # Pre-transform pipeline (ordered, conditional steps)
    pre_transform_pipeline_steps: List[str] = field(
        default_factory=lambda: ["analyze", "host_to_device", "evolve_markers", "warmup"]
    )
    pre_transform_warmup_model: Optional[str] = None   # Defaults to pre_transform_rewrite_model
    pre_transform_warmup_prompt: str = ""               # Uses pipeline default if empty
    pre_transform_marker_model: Optional[str] = None    # Defaults to pre_transform_rewrite_model

    # Dual pre-transform (LSA + GIN)
    pre_transform_dual: bool = False
    pre_transform_lsa_reference_code_path: Optional[str] = None
    pre_transform_lsa_nccl_api_docs: str = ""
    pre_transform_lsa_island_idx: int = 0
    pre_transform_gin_island_idx: int = 1


@dataclass
class RunningJob:
    """Represents a running job in the queue."""

    job_id: Union[str, Popen, ProcessWithLogging]
    exec_fname: str
    results_dir: str
    start_time: float
    generation: int
    parent_id: Optional[str]
    archive_insp_ids: List[str]
    top_k_insp_ids: List[str]
    code_diff: Optional[str]
    meta_patch_data: Optional[dict]
    code_embedding: List[float] = field(default_factory=list)
    embed_cost: float = 0.0
    novelty_cost: float = 0.0


# Set up logging
logger = logging.getLogger(__name__)


class EvolutionRunner:
    def __init__(
        self,
        evo_config: EvolutionConfig,
        job_config: JobConfig,
        db_config: DatabaseConfig,
        verbose: bool = True,
    ):
        self.evo_config = evo_config
        self.job_config = job_config
        self.db_config = db_config
        self.verbose = verbose

        print_gradient_logo((180, 50, 255), (50, 220, 255))
        if evo_config.results_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_dir = f"results_{timestamp}"
        else:
            self.results_dir = Path(evo_config.results_dir)

        if self.verbose:
            # Create log file path in results directory
            log_filename = f"{self.results_dir}/evolution_run.log"
            Path(self.results_dir).mkdir(parents=True, exist_ok=True)

            # Set up logging with both console and file handlers
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                handlers=[
                    RichHandler(
                        show_time=False, show_level=False, show_path=False
                    ),  # Console output (clean)
                    logging.FileHandler(
                        log_filename, mode="a", encoding="utf-8"
                    ),  # File output (detailed)
                ],
            )

            # Also log the initial setup information
            logger.info("=" * 80)
            start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Evolution run started at {start_time}")
            logger.info(f"Results directory: {self.results_dir}")
            logger.info(f"Log file: {log_filename}")
            logger.info("=" * 80)

        # Check if we are resuming a run
        resuming_run = False
        db_path = Path(f"{self.results_dir}/{db_config.db_path}")
        if self.evo_config.results_dir is not None and db_path.exists():
            resuming_run = True

        # Initialize LLM selection strategy
        if evo_config.llm_dynamic_selection is None:
            self.llm_selection = None
        elif isinstance(evo_config.llm_dynamic_selection, BanditBase):
            self.llm_selection = evo_config.llm_dynamic_selection
        elif (evo_config.llm_dynamic_selection.lower() == "ucb") or (
            evo_config.llm_dynamic_selection.lower() == "ucb1"
        ):
            self.llm_selection = AsymmetricUCB(
                arm_names=evo_config.llm_models,
                **evo_config.llm_dynamic_selection_kwargs,
            )
        else:
            raise ValueError("Invalid llm_dynamic_selection")

        # Initialize database and scheduler
        db_config.db_path = str(db_path)
        embedding_model_to_use = (
            evo_config.embedding_model or "text-embedding-3-small"
        )
        self.db = ProgramDatabase(
            config=db_config, embedding_model=embedding_model_to_use
        )
        self.scheduler = JobScheduler(
            job_type=evo_config.job_type,
            config=job_config,  # type: ignore
            verbose=verbose,
        )

        self.llm = LLMClient(
            model_names=evo_config.llm_models,
            model_selection=self.llm_selection,
            **evo_config.llm_kwargs,
            verbose=verbose,
        )
        if evo_config.embedding_model is not None:
            self.embedding = EmbeddingClient(
                model_name=evo_config.embedding_model,
                verbose=verbose,
            )
        else:
            self.embedding = None

        if evo_config.meta_llm_models is not None:
            self.meta_llm = LLMClient(
                model_names=evo_config.meta_llm_models,
                **evo_config.meta_llm_kwargs,
                verbose=verbose,
            )
        else:
            self.meta_llm = None

        if evo_config.novelty_llm_models is not None:
            self.novelty_llm = LLMClient(
                model_names=evo_config.novelty_llm_models,
                **evo_config.novelty_llm_kwargs,
                verbose=verbose,
            )
        else:
            self.novelty_llm = None

        # Initialize PromptSampler for handling LLM code prompts
        self.prompt_sampler = PromptSampler(
            task_sys_msg=evo_config.task_sys_msg,
            task_sys_msg_per_island=evo_config.task_sys_msg_per_island,
            language=evo_config.language,
            patch_types=evo_config.patch_types,
            patch_type_probs=evo_config.patch_type_probs,
            use_text_feedback=evo_config.use_text_feedback,
        )

        # Initialize MetaSummarizer for meta-recommendations
        self.meta_summarizer = MetaSummarizer(
            meta_llm_client=self.meta_llm,
            language=evo_config.language,
            use_text_feedback=evo_config.use_text_feedback,
            max_recommendations=evo_config.meta_max_recommendations,
        )

        # Initialize NoveltyJudge for novelty assessment
        self.novelty_judge = NoveltyJudge(
            novelty_llm_client=self.novelty_llm,
            language=evo_config.language,
            similarity_threshold=evo_config.code_embed_sim_threshold,
            max_novelty_attempts=evo_config.max_novelty_attempts,
        )

        # Initialize rich console for formatted output
        self.console = Console()

        if self.evo_config.language == "cuda":
            self.lang_ext = "cu"
        elif self.evo_config.language == "cpp":
            self.lang_ext = "cpp"
        elif self.evo_config.language == "python":
            self.lang_ext = "py"
        elif self.evo_config.language == "rust":
            self.lang_ext = "rs"
        elif self.evo_config.language == "swift":
            self.lang_ext = "swift"
        elif self.evo_config.language in ["json", "json5"]:
            self.lang_ext = "json"
        else:
            msg = f"Language {self.evo_config.language} not supported"
            raise ValueError(msg)

        # Queue for managing parallel jobs
        self.running_jobs: List[RunningJob] = []
        self.best_program_id: Optional[str] = None
        self.next_generation_to_submit = 0

        if resuming_run:
            self.completed_generations = self.db.last_iteration + 1
            self.next_generation_to_submit = self.completed_generations
            logger.info("=" * 80)
            logger.info("RESUMING PREVIOUS EVOLUTION RUN")
            logger.info("=" * 80)
            logger.info(
                f"Resuming evolution from: {self.results_dir}\n"
                f"Found {self.completed_generations} "
                "previously completed generations."
            )
            logger.info("=" * 80)
            self._update_best_solution()
            # Restore meta memory state when resuming
            self._restore_meta_memory()
        else:
            self.completed_generations = 0

        # Save experiment configuration to a YAML file
        self._save_experiment_config(evo_config, job_config, db_config)

    def _save_experiment_config(
        self,
        evo_config: EvolutionConfig,
        job_config: JobConfig,
        db_config: DatabaseConfig,
    ) -> None:
        """Save experiment configuration to a YAML file."""
        evo_dict = asdict(evo_config)
        for key in list(evo_dict):
            val = evo_dict[key]
            if callable(val):
                evo_dict[key] = (
                    f"<callable: {val.__name__}>"
                    if hasattr(val, "__name__") else "<callable>"
                )
            elif val is not None and not isinstance(val, (str, int, float, bool, list, dict)):
                evo_dict[key] = str(val)
        config_data = {
            "evolution_config": evo_dict,
            "job_config": asdict(job_config),
            "database_config": asdict(db_config),
            "timestamp": datetime.now().isoformat(),
            "results_directory": str(self.results_dir),
        }

        config_path = Path(self.results_dir) / "experiment_config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with config_path.open("w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)

        logger.info(f"Experiment configuration saved to {config_path}")

    def _run_pre_transform(self) -> None:
        """Run pre-transform pipeline on init_program_path.

        Dispatch order:
          1. Agent mode (``pre_transform_agent``) — Claude Code with full autonomy
          2. Pipeline mode (``pre_transform_pipeline_steps``) — ordered conditional steps

        On success, updates self.evo_config.init_program_path to point to the
        transformed file so that generation 0 starts from the prepared code.
        """
        source_path = Path(self.evo_config.init_program_path).resolve()
        if not source_path.exists():
            logger.warning(
                f"Pre-transform: init_program_path not found: {source_path}. Skipping."
            )
            return

        if self.evo_config.pre_transform_agent:
            self._run_pre_transform_agent(source_path)
        elif self.evo_config.pre_transform_pipeline_steps:
            self._run_pre_transform_pipeline(source_path)
        else:
            logger.warning(
                "Pre-transform enabled but no agent or pipeline steps configured. Skipping."
            )

    # ------------------------------------------------------------------
    # Agent-based pre-transform (Claude Code with full tool access)
    # ------------------------------------------------------------------

    def _run_pre_transform_agent(self, source_path: Path) -> None:
        """Use Claude Code CLI as an autonomous agent for pre-transform.

        The agent gets Bash/Read/Write/Edit tools and iterates freely until
        the transformation succeeds or the budget is exhausted.
        """
        import os
        import subprocess as _sp

        from cuco.transform import insert_evolve_markers

        output_dir = Path(self.results_dir) / "_pre_transform"
        output_dir.mkdir(parents=True, exist_ok=True)
        work_dir = source_path.parent / "_transform_work"
        work_dir.mkdir(parents=True, exist_ok=True)

        binary_name = source_path.stem
        output_file = output_dir / f"{binary_name}_device.cu"

        # Build the prompt — use custom builder if provided, else generic one
        prompt_builder = self.evo_config.pre_transform_agent_prompt_builder
        if callable(prompt_builder):
            prompt = prompt_builder(source_path, work_dir)
        else:
            prompt = self._build_default_agent_prompt(source_path, work_dir)

        system_prompt = (
            self.evo_config.pre_transform_agent_system_prompt
            or self._default_agent_system_prompt()
        )

        (work_dir / "_agent_prompt.txt").write_text(prompt, encoding="utf-8")

        model = self.evo_config.pre_transform_agent_model

        cmd = [
            "claude", "-p",
            "--model", model,
            "--output-format", "text",
            "--dangerously-skip-permissions",
            "--no-session-persistence",
            "--allowedTools", "Bash,Read,Write,Edit",
            "--add-dir", str(work_dir),
            "--add-dir", str(source_path.parent),
            "--system-prompt", system_prompt,
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
        for p in ["/usr/bin", "/usr/local/bin", "/usr/sbin",
                  "/usr/local/cuda-13.1/bin", os.path.expanduser("~/.local/bin")]:
            if p not in env.get("PATH", ""):
                env["PATH"] = p + ":" + env.get("PATH", "")

        logger.info(f"Pre-transform AGENT mode: model={model}")
        start = time.perf_counter()
        try:
            result = _sp.run(
                cmd, input=prompt, cwd=str(work_dir),
                capture_output=True, text=True, timeout=1800, env=env,
            )
        except _sp.TimeoutExpired:
            logger.error("Pre-transform agent timed out (30 min)")
            return
        except Exception as exc:
            logger.error(f"Pre-transform agent failed: {exc}")
            return

        duration = time.perf_counter() - start
        agent_output = (result.stdout or "") + "\n" + (result.stderr or "")
        (work_dir / "_agent_output.txt").write_text(agent_output, encoding="utf-8")
        logger.info(
            f"Pre-transform agent finished in {duration:.1f}s  "
            f"exit={result.returncode}"
        )

        # Find the transformed file
        working_file = work_dir / f"{binary_name}.cu"
        device_file = work_dir / f"{binary_name}_device.cu"
        final_code = ""
        if device_file.exists():
            final_code = device_file.read_text(encoding="utf-8")
        elif working_file.exists():
            final_code = working_file.read_text(encoding="utf-8")

        if not final_code:
            logger.warning(
                "Pre-transform agent produced no output file. "
                f"Using original code: {source_path}"
            )
            return

        marker_model = f"claude-cli/{model}"
        marked_code = insert_evolve_markers(final_code, llm_model=marker_model)
        output_file.write_text(marked_code, encoding="utf-8")

        logger.info(
            f"Pre-transform AGENT SUCCEEDED in {duration:.1f}s. "
            f"Using transformed code: {output_file}"
        )
        self.evo_config.init_program_path = str(output_file)

    @staticmethod
    def _default_agent_system_prompt() -> str:
        return (
            "You are an expert CUDA and NCCL programmer specializing in "
            "GPU-Initiated Networking (GIN). You transform CUDA programs from "
            "host-side NCCL collectives to device-side GIN.\n\n"
            "Be methodical:\n"
            "1. Read and understand the source code first\n"
            "2. Plan the transformation\n"
            "3. Implement step by step\n"
            "4. Build, test, and fix any issues\n"
            "5. Do not give up — keep iterating until verification passes\n\n"
            "When you encounter errors:\n"
            "- Read error messages carefully\n"
            "- Check for common GIN pitfalls (wrong Coop for flush, missing "
            "__syncthreads, signal count mismatch)\n"
            "- Use the provided API reference to verify correct usage\n"
            "- Add debug prints if needed for runtime diagnosis"
        )

    def _build_default_agent_prompt(
        self, source_path: Path, work_dir: Path
    ) -> str:
        """Build a generic agent prompt from config and NCCL docs."""
        binary_name = source_path.stem
        output_file = work_dir / f"{binary_name}_device.cu"
        nccl_docs = self.evo_config.pre_transform_nccl_api_docs

        parts: list[str] = [
            f"## Task: Transform CUDA Host-Side NCCL → Device-Side GIN\n\n"
            f"Transform the CUDA source file at `{source_path}` so that all "
            f"host-side NCCL collective calls (ncclAllReduce, ncclAllGather, "
            f"etc.) are replaced with device-side GIN communication "
            f"(gin.put / gin.flush / gin.waitSignal).\n\n"
            f"### Working directory\n"
            f"All intermediate files go in: `{work_dir}`\n"
            f"Write the working copy as: `{work_dir}/{binary_name}.cu`\n"
            f"Write the final verified version to: `{output_file}`\n\n"
            f"### Success criteria\n"
            f"1. Code compiles without errors\n"
            f"2. Program runs successfully with mpirun -np 4\n"
            f"3. Program prints \"Verification: PASS\"\n"
            f"4. **No remaining host-side NCCL collective calls** — all "
            f"communication uses GIN\n\n"
            f"### CRITICAL: clear stale CUDA error after ncclDevCommCreate\n\n"
            f"```cpp\n"
            f"NCCL_CHECK(ncclDevCommCreate(host_comm, &reqs, &devComm));\n"
            f"(void)cudaGetLastError();  // MUST clear stale CUDA error\n"
            f"```\n",
        ]

        if nccl_docs:
            parts.append(
                "\n### NCCL Device API Reference\n\n" + nccl_docs
            )

        parts.append(
            f"\n### Instructions\n"
            f"1. Read the source file at `{source_path}`\n"
            f"2. Copy it to `{work_dir}/{binary_name}.cu` and start "
            f"transforming\n"
            f"3. After each stage, build and run\n"
            f"4. If build or run fails, read the errors, fix, and retry\n"
            f"5. Once \"Verification: PASS\" appears and no host-side "
            f"collectives remain, write the final code to `{output_file}`"
        )

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Pipeline-based pre-transform
    # ------------------------------------------------------------------

    def _run_pre_transform_pipeline(self, source_path: Path) -> None:
        """Run the multi-step pre-transform pipeline.

        Steps are configured via ``pre_transform_pipeline_steps`` and run in
        order.  Each step checks whether it is needed (regex/Python) and only
        applies when necessary (LLM or regex).
        """
        from cuco.transform.pipeline import PreTransformPipeline
        from cuco.transform.transformer import TransformConfig

        rewrite_model = self.evo_config.pre_transform_rewrite_model
        if rewrite_model is None and self.evo_config.llm_models:
            rewrite_model = self.evo_config.llm_models[0]
        if rewrite_model is None:
            rewrite_model = "bedrock/us.anthropic.claude-sonnet-4-6"

        reference_code = ""
        if self.evo_config.pre_transform_reference_code_path:
            ref_path = Path(self.evo_config.pre_transform_reference_code_path)
            if ref_path.exists():
                reference_code = ref_path.read_text(encoding="utf-8")
                logger.info(f"Pipeline: Loaded reference code from {ref_path}")

        config = TransformConfig(
            rewrite_model=rewrite_model,
            judge_model=self.evo_config.pre_transform_judge_model,
            max_iterations=self.evo_config.pre_transform_max_iterations,
            two_stage=self.evo_config.pre_transform_two_stage,
            stage_a_max_iterations=self.evo_config.pre_transform_stage_a_max_iterations,
            stage_b_max_iterations=self.evo_config.pre_transform_stage_b_max_iterations,
            reference_code=reference_code,
            nccl_api_docs=self.evo_config.pre_transform_nccl_api_docs,
            binary_name=source_path.stem,
        )

        warmup_model = self.evo_config.pre_transform_warmup_model or rewrite_model
        marker_model = self.evo_config.pre_transform_marker_model or rewrite_model

        output_dir = Path(self.results_dir) / "_pre_transform_pipeline"
        work_dir = source_path.parent / "_transform_work"

        pipeline = PreTransformPipeline(
            config=config,
            steps=self.evo_config.pre_transform_pipeline_steps,
            warmup_model=warmup_model,
            warmup_prompt=self.evo_config.pre_transform_warmup_prompt,
            marker_model=marker_model,
        )

        result = pipeline.run(source_path, output_dir, work_dir=work_dir)

        if not hasattr(self, "_pre_transform_costs"):
            self._pre_transform_costs = {}
        self._pre_transform_costs[0] = result.total_cost
        logger.info(f"Pipeline cost: ${result.total_cost:.4f}")

        step_summary = ", ".join(
            f"{sr.name}={'skip' if sr.skipped else ('err' if sr.error else 'ok')}"
            for sr in result.step_results
        )
        logger.info(f"Pipeline steps: [{step_summary}]")

        if result.final_path and not result.any_step_failed:
            logger.info(
                f"Pre-transform pipeline SUCCEEDED ({result.total_duration_sec:.1f}s). "
                f"Using: {result.final_path}"
            )
            self.evo_config.init_program_path = result.final_path
        elif result.final_path:
            logger.warning(
                f"Pre-transform pipeline completed WITH ERRORS ({result.total_duration_sec:.1f}s). "
                f"Using best-effort output: {result.final_path}"
            )
            self.evo_config.init_program_path = result.final_path
        else:
            logger.warning(
                f"Pre-transform pipeline FAILED ({result.total_duration_sec:.1f}s). "
                f"Using original code: {source_path}"
            )

    def _run_pre_transform_dual(self) -> None:
        """Run dual pre-transform: produce both an LSA and a GIN seed from host code.

        On success, populates init_program_paths_per_island for all islands
        sharing each API type (e.g. both LSA islands get the LSA seed).
        """
        from cuco.transform import CUDAAnalyzer, HostToDeviceTransformer
        from cuco.transform.transformer import TransformConfig

        source_path = Path(self.evo_config.init_program_path).resolve()
        if not source_path.exists():
            logger.warning(
                f"Dual pre-transform: init_program_path not found: {source_path}. Skipping."
            )
            return

        analyzer = CUDAAnalyzer(source_path)
        report = analyzer.analyze()

        if not report.has_host_communication():
            logger.info(
                "Dual pre-transform: No host-side NCCL collectives found. Skipping."
            )
            return

        rewrite_model = self.evo_config.pre_transform_rewrite_model
        if rewrite_model is None and self.evo_config.llm_models:
            rewrite_model = self.evo_config.llm_models[0]
        if rewrite_model is None:
            rewrite_model = "bedrock/us.anthropic.claude-sonnet-4-6"

        output_dir = Path(self.results_dir) / "_pre_transform"
        output_dir.mkdir(parents=True, exist_ok=True)

        transforms = [
            {
                "api": "gin",
                "ref_path": self.evo_config.pre_transform_reference_code_path,
                "api_docs": self.evo_config.pre_transform_nccl_api_docs,
                "island_idx": self.evo_config.pre_transform_gin_island_idx,
                "seed_name": "gin_seed",
            },
            {
                "api": "lsa",
                "ref_path": self.evo_config.pre_transform_lsa_reference_code_path,
                "api_docs": self.evo_config.pre_transform_lsa_nccl_api_docs,
                "island_idx": self.evo_config.pre_transform_lsa_island_idx,
                "seed_name": "lsa_seed",
            },
        ]

        if self.evo_config.init_program_paths_per_island is None:
            self.evo_config.init_program_paths_per_island = {}

        if not hasattr(self, '_pre_transform_costs'):
            self._pre_transform_costs: Dict[int, float] = {}

        for t in transforms:
            logger.info(f"Dual pre-transform: Running {t['api'].upper()} transform...")

            reference_code = ""
            if t["ref_path"]:
                ref_p = Path(t["ref_path"])
                if ref_p.exists():
                    reference_code = ref_p.read_text(encoding="utf-8")
                    logger.info(
                        f"  Loaded {t['api'].upper()} reference from {ref_p}"
                    )

            config = TransformConfig(
                rewrite_model=rewrite_model,
                judge_model=self.evo_config.pre_transform_judge_model,
                max_iterations=self.evo_config.pre_transform_max_iterations,
                two_stage=self.evo_config.pre_transform_two_stage,
                stage_a_max_iterations=self.evo_config.pre_transform_stage_a_max_iterations,
                stage_b_max_iterations=self.evo_config.pre_transform_stage_b_max_iterations,
                api_type=t["api"],
                reference_code=reference_code,
                nccl_api_docs=t["api_docs"],
                binary_name=source_path.stem,
            )

            work_dir = source_path.parent / f"_transform_work_{t['api']}"
            api_output_dir = output_dir / t["api"]

            transformer = HostToDeviceTransformer(config)
            result = transformer.transform(source_path, work_dir=work_dir)
            transformer.save_result(result, api_output_dir)

            island_idx = t["island_idx"]
            self._pre_transform_costs[island_idx] = result.total_cost
            logger.info(
                f"  {t['api'].upper()} transform cost: ${result.total_cost:.4f}"
            )

            if result.success:
                seed_path = str(api_output_dir / result.device_filename)
                self.evo_config.init_program_paths_per_island[island_idx] = seed_path
                # Also seed any other islands with the same API type
                api_types = getattr(self.db_config, "island_api_types", None)
                if api_types:
                    for idx, api_type in api_types.items():
                        if api_type == t["api"] and idx != island_idx:
                            self.evo_config.init_program_paths_per_island[idx] = seed_path
                            logger.info(
                                f"  Also seeding island {idx} ({api_type}) with same seed"
                            )
                logger.info(
                    f"  {t['api'].upper()} transform SUCCEEDED -> {seed_path}"
                )
            else:
                logger.warning(
                    f"  {t['api'].upper()} transform FAILED after "
                    f"{len(result.iterations)} iterations."
                )
                if result.error:
                    logger.warning(f"  Error: {result.error[:300]}")

    def run(self):
        """Run evolution with parallel job queue."""
        max_jobs = self.evo_config.max_parallel_jobs
        target_gens = self.evo_config.num_generations
        logger.info(
            f"Starting evolution with {max_jobs} parallel jobs, "
            f"target: {target_gens} generations"
        )

        # Run pre-transform pipeline if enabled
        if self.evo_config.pre_transform_enabled and self.completed_generations == 0:
            logger.info("=" * 60)
            if self.evo_config.pre_transform_dual:
                logger.info("DUAL PRE-TRANSFORM: Producing both LSA and GIN seeds...")
                logger.info("=" * 60)
                self._run_pre_transform_dual()
            else:
                steps = self.evo_config.pre_transform_pipeline_steps
                if steps:
                    logger.info(
                        f"PRE-TRANSFORM PIPELINE: steps={steps}"
                    )
                else:
                    logger.info("PRE-TRANSFORM: Checking for host-side NCCL collectives...")
                logger.info("=" * 60)
                self._run_pre_transform()
            logger.info("=" * 60)

        # First, run generation 0 sequentially to populate the database
        if self.completed_generations == 0 and target_gens > 0:
            logger.info("Running generation 0 sequentially to initialize database...")
            if self.evo_config.init_program_paths_per_island:
                self._run_generation_0_multi_seed()
            else:
                self._run_generation_0()
            self.completed_generations = 1
            self.next_generation_to_submit = 1
            logger.info(f"Completed generation 0, total: 1/{target_gens}")

        # Now start parallel execution for remaining generations
        if self.completed_generations < target_gens:
            logger.info("Starting parallel execution for remaining generations...")

            # Main loop: monitor jobs and submit new ones
            while (
                self.completed_generations < target_gens or len(self.running_jobs) > 0
            ):
                # Check for completed jobs
                completed_jobs = self._check_completed_jobs()

                # Process completed jobs
                if completed_jobs:
                    for job in completed_jobs:
                        self._process_completed_job(job)

                    # Update completed generations count
                    self._update_completed_generations()

                    if self.verbose:
                        logger.info(
                            f"Processed {len(completed_jobs)} jobs. "
                            f"Total completed generations: "
                            f"{self.completed_generations}/{target_gens}"
                        )

                # Check if we've completed all generations
                if self.completed_generations >= target_gens:
                    logger.info("All generations completed, exiting...")
                    break

                # Submit new jobs to fill the queue (only if we have capacity)
                if (
                    len(self.running_jobs) < max_jobs
                    and self.next_generation_to_submit < target_gens
                ):
                    self._submit_new_job()

                # Wait a bit before checking again
                time.sleep(2)

            # All jobs are now handled by the main loop above

        # Perform final meta summary for any remaining unprocessed programs
        best_program = self.db.get_best_program()
        self.meta_summarizer.perform_final_summary(str(self.results_dir), best_program)

        # Save final meta memory state
        self._save_meta_memory()

        self.db.print_summary()
        logger.info(f"Evolution completed! {self.completed_generations} generations")
        logger.info("=" * 80)
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Evolution run ended at {end_time}")
        logger.info("=" * 80)

    def generate_initial_program(self):
        """Generate initial program with LLM, with retries."""
        llm_kwargs = self.llm.get_kwargs()

        sys_msg, user_msg = self.prompt_sampler.initial_program_prompt()
        msg_history = []
        total_costs = 0.0

        for attempt in range(self.evo_config.max_patch_attempts):
            response = self.llm.query(
                msg=user_msg,
                system_msg=sys_msg,
                llm_kwargs=llm_kwargs,
                msg_history=msg_history,
            )
            if response is None or response.content is None:
                if self.verbose:
                    logger.info(
                        f"  INITIAL PROGRAM ATTEMPT {attempt + 1}/"
                        f"{self.evo_config.max_patch_attempts} "
                        "FAILURE. Error: LLM response content was None."
                    )
                if attempt < self.evo_config.max_patch_attempts - 1:
                    user_msg = (
                        "The previous response was empty. Please try again "
                        "and provide the full code."
                    )
                    if response and response.new_msg_history:
                        msg_history = response.new_msg_history
                    continue
                else:
                    break

            total_costs += response.cost or 0
            initial_code = extract_between(
                response.content,
                f"```{self.evo_config.language}",
                "```",
                False,
            )

            if initial_code:
                patch_name = extract_between(
                    response.content, "<NAME>", "</NAME>", False
                )
                patch_description = extract_between(
                    response.content, "<DESCRIPTION>", "</DESCRIPTION>", False
                )
                if self.evo_config.language == "python":
                    comment_char = "#"
                else:
                    comment_char = "//"

                initial_code = (
                    f"{comment_char} EVOLVE-BLOCK-START\n"
                    f"{initial_code}\n"
                    f"{comment_char} EVOLVE-BLOCK-END\n"
                )

                if self.verbose:
                    logger.info(
                        f"  INITIAL PROGRAM ATTEMPT {attempt + 1}/"
                        f"{self.evo_config.max_patch_attempts} "
                        "SUCCESS."
                    )
                return initial_code, patch_name, patch_description, total_costs
            else:  # code extraction failed
                if self.verbose:
                    logger.info(
                        f"  INITIAL PROGRAM ATTEMPT {attempt + 1}/"
                        f"{self.evo_config.max_patch_attempts} "
                        "FAILURE. Error: Could not extract code from response."
                    )
                if attempt < self.evo_config.max_patch_attempts - 1:
                    user_msg = (
                        "Could not extract code from your last response. "
                        "Please make sure to enclose the code in "
                        "`<CODE>`...`</CODE>` tags."
                    )
                    msg_history = response.new_msg_history
                else:  # last attempt
                    break

        raise ValueError(
            "LLM failed to generate a valid initial program after "
            f"{self.evo_config.max_patch_attempts} attempts."
        )

    def _run_generation_0(self):
        """Setup and run generation 0 to initialize the database."""
        initial_dir = f"{self.results_dir}/{FOLDER_PREFIX}_0"
        Path(initial_dir).mkdir(parents=True, exist_ok=True)
        exec_fname = f"{initial_dir}/main.{self.lang_ext}"
        results_dir = f"{self.results_dir}/{FOLDER_PREFIX}_0/results"
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        api_costs = getattr(self, '_pre_transform_costs', {}).get(0, 0.0)
        patch_name = "initial_program"
        patch_description = "Initial program from file."
        patch_type = "init"

        if self.evo_config.init_program_path:
            if self.verbose:
                logger.info(
                    f"Copying initial program from {self.evo_config.init_program_path}"
                )
            shutil.copy(self.evo_config.init_program_path, exec_fname)
        else:
            if self.verbose:
                logger.info(
                    "`init_program_path` not provided, "
                    "generating initial program with LLM..."
                )
            initial_code, patch_name, patch_description, api_costs = (
                self.generate_initial_program()
            )
            with open(exec_fname, "w", encoding="utf-8") as f:
                f.write(initial_code)

            if self.verbose:
                logger.info(f"Initial program generated and saved to {exec_fname}")

        # Run the evaluation synchronously
        results, rtime = self.scheduler.run(exec_fname, results_dir)

        code_embedding, e_cost = self.get_code_embedding(exec_fname)

        # Read the evaluated code for database insertion
        try:
            evaluated_code = Path(exec_fname).read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not read code for job {exec_fname}. Error: {e}")
            evaluated_code = ""

        correct_val = False
        metrics_val = {}
        stdout_log = ""
        stderr_log = ""
        if results:
            correct_val = results.get("correct", {}).get("correct", False)
            metrics_val = results.get("metrics", {})
            stdout_log = results.get("stdout_log", "")
            stderr_log = results.get("stderr_log", "")

        combined_score = metrics_val.get("combined_score", 0.0)
        public_metrics = metrics_val.get("public", {})
        private_metrics = metrics_val.get("private", {})
        text_feedback = metrics_val.get("text_feedback", "")

        # Add the program to the database
        db_program = Program(
            id=str(uuid.uuid4()),
            code=evaluated_code,
            language=self.evo_config.language,
            parent_id=None,
            generation=0,
            archive_inspiration_ids=[],
            top_k_inspiration_ids=[],
            code_diff=None,
            embedding=code_embedding,
            correct=correct_val,
            combined_score=combined_score,
            public_metrics=public_metrics,
            private_metrics=private_metrics,
            text_feedback=text_feedback,
            metadata={
                "compute_time": rtime,
                "api_costs": api_costs,
                "embed_cost": e_cost,
                "novelty_cost": 0.0,  # No novelty cost for generation 0
                "patch_type": patch_type,
                "patch_name": patch_name,
                "patch_description": patch_description,
                "stdout_log": stdout_log,
                "stderr_log": stderr_log,
            },
        )

        self.db.add(db_program, verbose=True)
        if self.llm_selection is not None:
            self.llm_selection.set_baseline_score(
                db_program.combined_score if correct_val else 0.0,
            )
        self.db.save()
        self._update_best_solution()

        # Add the evaluated program to meta memory tracking
        self.meta_summarizer.add_evaluated_program(db_program)

        # Check if we should update meta memory after adding this program
        if self.meta_summarizer.should_update_meta(self.evo_config.meta_rec_interval):
            logger.info(
                f"Updating meta memory after processing "
                f"{len(self.meta_summarizer.evaluated_since_last_meta)} programs..."
            )
            best_program = self.db.get_best_program()
            updated_recs, meta_cost = self.meta_summarizer.update_meta_memory(
                best_program
            )
            if updated_recs:
                # Write meta output file for generation 0
                self.meta_summarizer.write_meta_output(str(self.results_dir))
                # Store meta cost for tracking
                if meta_cost > 0:
                    logger.info(
                        f"Meta recommendation generation cost: ${meta_cost:.4f}"
                    )
                    # Add meta cost to this program's metadata (the one that triggered the update)
                    if db_program.metadata is None:
                        db_program.metadata = {}
                    db_program.metadata["meta_cost"] = meta_cost
                    # Update the program in the database with the new metadata
                    import json

                    metadata_json = json.dumps(db_program.metadata)
                    self.db.cursor.execute(
                        "UPDATE programs SET metadata = ? WHERE id = ?",
                        (metadata_json, db_program.id),
                    )
                    self.db.conn.commit()

        # Save meta memory state after each job completion
        self._save_meta_memory()

    def _run_generation_0_for_island(
        self, island_idx: int, init_program_path: str
    ) -> Program:
        """Run generation 0 for a specific island with a specific seed program."""
        island_dir = f"{self.results_dir}/{FOLDER_PREFIX}_0_island{island_idx}"
        Path(island_dir).mkdir(parents=True, exist_ok=True)
        exec_fname = f"{island_dir}/main.{self.lang_ext}"
        results_dir = f"{island_dir}/results"
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Seeding island {island_idx} from {init_program_path}"
        )
        shutil.copy(init_program_path, exec_fname)

        results, rtime = self.scheduler.run(exec_fname, results_dir)
        code_embedding, e_cost = self.get_code_embedding(exec_fname)

        try:
            evaluated_code = Path(exec_fname).read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not read code for {exec_fname}: {e}")
            evaluated_code = ""

        correct_val = False
        metrics_val = {}
        stdout_log = ""
        stderr_log = ""
        if results:
            correct_val = results.get("correct", {}).get("correct", False)
            metrics_val = results.get("metrics", {})
            stdout_log = results.get("stdout_log", "")
            stderr_log = results.get("stderr_log", "")

        combined_score = metrics_val.get("combined_score", 0.0)
        public_metrics = metrics_val.get("public", {})
        private_metrics = metrics_val.get("private", {})
        text_feedback = metrics_val.get("text_feedback", "")

        db_program = Program(
            id=str(uuid.uuid4()),
            code=evaluated_code,
            language=self.evo_config.language,
            parent_id=None,
            generation=0,
            archive_inspiration_ids=[],
            top_k_inspiration_ids=[],
            code_diff=None,
            embedding=code_embedding,
            correct=correct_val,
            combined_score=combined_score,
            public_metrics=public_metrics,
            private_metrics=private_metrics,
            text_feedback=text_feedback,
            island_idx=island_idx,
            metadata={
                "compute_time": rtime,
                "api_costs": getattr(self, '_pre_transform_costs', {}).get(island_idx, 0.0),
                "embed_cost": e_cost,
                "novelty_cost": 0.0,
                "patch_type": "init",
                "patch_name": f"island_{island_idx}_seed",
                "patch_description": f"Seed program for island {island_idx}",
                "stdout_log": stdout_log,
                "stderr_log": stderr_log,
                "source_dir": island_dir,
            },
        )

        self.db.add(db_program, verbose=True)
        self.db.save()
        self.meta_summarizer.add_evaluated_program(db_program)
        return db_program

    def _run_generation_0_multi_seed(self):
        """Multi-seed generation 0: each island gets its own initial program.
        Islands not in init_program_paths_per_island get seeded with the
        best-scoring seed from the other islands."""
        paths_per_island = self.evo_config.init_program_paths_per_island
        if not paths_per_island:
            return self._run_generation_0()

        num_islands = self.db_config.num_islands
        seeded_programs = {}

        for island_idx, path in paths_per_island.items():
            prog = self._run_generation_0_for_island(island_idx, path)
            seeded_programs[island_idx] = prog

        # Seed remaining islands with the best-scoring seed
        if seeded_programs:
            best_seed = max(
                seeded_programs.values(),
                key=lambda p: p.combined_score if p.correct else -1e9,
            )
            for island_idx in range(num_islands):
                if island_idx not in seeded_programs:
                    seed_path = self.evo_config.init_program_paths_per_island.get(
                        best_seed.island_idx
                    )
                    if seed_path:
                        prog = self._run_generation_0_for_island(island_idx, seed_path)
                        seeded_programs[island_idx] = prog

        if self.llm_selection is not None:
            best = max(
                seeded_programs.values(),
                key=lambda p: p.combined_score if p.correct else 0.0,
            )
            self.llm_selection.set_baseline_score(
                best.combined_score if best.correct else 0.0,
            )

        self._update_best_solution()
        self._save_meta_memory()

    def _update_completed_generations(self):
        """
        Update the count of completed generations from the database.
        A generation `g` is considered complete if all generations from 0..g
        have at least one program in the database. This ensures the count
        advances sequentially without gaps.
        """
        last_gen = self.db.last_iteration
        if last_gen == -1:
            self.completed_generations = 0
            return

        # Check for contiguous generations from 0 up to last_gen
        completed_up_to = 0
        for i in range(last_gen + 1):
            if self.db.get_programs_by_generation(i):
                completed_up_to = i + 1
            else:
                # Found a gap, so contiguous sequence is broken
                self.completed_generations = completed_up_to
                return

        self.completed_generations = completed_up_to

    def _submit_new_job(self):
        """Submit a new job to the queue."""
        current_gen = self.next_generation_to_submit

        if current_gen >= self.evo_config.num_generations:
            return

        self.next_generation_to_submit += 1

        exec_fname = (
            f"{self.results_dir}/{FOLDER_PREFIX}_{current_gen}/main.{self.lang_ext}"
        )
        results_dir = f"{self.results_dir}/{FOLDER_PREFIX}_{current_gen}/results"
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        # Get current meta-recommendations for this job
        meta_recs, meta_summary, meta_scratch = self.meta_summarizer.get_current()

        # Sample parent and inspiration programs
        if current_gen == 0:
            parent_id = None
            archive_insp_ids = []
            top_k_insp_ids = []
            code_diff = None
            meta_patch_data = {}
            # Initial program already copied in setup_initial_program
        else:
            api_costs = 0
            embed_cost = 0
            novelty_cost = 0.0
            novelty_checks_performed = 0
            # Loop over novelty attempts
            for nov_attempt in range(self.evo_config.max_novelty_attempts):
                # Loop over patch resamples - including parents
                for resample in range(self.evo_config.max_patch_resamples):
                    (
                        parent_program,
                        archive_programs,
                        top_k_programs,
                    ) = self.db.sample(
                        target_generation=current_gen,
                        novelty_attempt=nov_attempt + 1,
                        max_novelty_attempts=self.evo_config.max_novelty_attempts,
                        resample_attempt=resample + 1,
                        max_resample_attempts=self.evo_config.max_patch_resamples,
                    )
                    archive_insp_ids = [p.id for p in archive_programs]
                    top_k_insp_ids = [p.id for p in top_k_programs]
                    parent_id = parent_program.id
                    # Run patch (until success with max attempts)
                    code_diff, meta_patch_data, num_applied_attempt = self.run_patch(
                        parent_program,
                        archive_programs,
                        top_k_programs,
                        current_gen,
                        novelty_attempt=nov_attempt + 1,
                        resample_attempt=resample + 1,
                        island_idx=parent_program.island_idx,
                    )
                    api_costs += meta_patch_data["api_costs"]
                    if (
                        meta_patch_data["error_attempt"] is None
                        and num_applied_attempt > 0
                    ):
                        meta_patch_data["api_costs"] = api_costs
                        break

                # Get the code embedding for the evaluated code
                code_embedding, e_cost = self.get_code_embedding(exec_fname)
                embed_cost += e_cost

                if not code_embedding:
                    self.novelty_judge.log_novelty_skip_message("no embedding")
                    break

                # Use NoveltyJudge for novelty assessment with rejection sampling
                if self.novelty_judge.should_check_novelty(
                    code_embedding, current_gen, parent_program, self.db
                ):
                    should_accept, novelty_metadata = (
                        self.novelty_judge.assess_novelty_with_rejection_sampling(
                            exec_fname, code_embedding, parent_program, self.db
                        )
                    )

                    # Update costs and metadata from novelty assessment
                    novelty_cost += novelty_metadata.get("novelty_total_cost", 0.0)
                    novelty_checks_performed = novelty_metadata.get(
                        "novelty_checks_performed", 0
                    )
                    novelty_explanation = novelty_metadata.get(
                        "novelty_explanation", ""
                    )

                    if should_accept:
                        break
                    # If not accepted, continue to next attempt (rejection sampling)
                else:
                    if not self.db.island_manager or not hasattr(
                        self.db.island_manager, "are_all_islands_initialized"
                    ):
                        self.novelty_judge.log_novelty_skip_message("no island manager")
                    elif not self.db.island_manager.are_all_islands_initialized():
                        self.novelty_judge.log_novelty_skip_message(
                            "not all islands initialized yet"
                        )
                    break

        # Add meta-recommendations/summary/scratchpad to meta_patch_data
        if meta_recs is not None:
            meta_patch_data["meta_recommendations"] = meta_recs
            meta_patch_data["meta_summary"] = meta_summary
            meta_patch_data["meta_scratch_pad"] = meta_scratch

        # Add novelty check information to meta_patch_data if any checks were performed
        if current_gen > 0 and novelty_checks_performed > 0:
            meta_patch_data["novelty_checks_performed"] = novelty_checks_performed
            meta_patch_data["novelty_cost"] = novelty_cost
            meta_patch_data["novelty_explanation"] = novelty_explanation

        # Submit the job asynchronously
        job_id = self.scheduler.submit_async(exec_fname, results_dir)

        # Add to running jobs queue
        running_job = RunningJob(
            job_id=job_id,
            exec_fname=exec_fname,
            results_dir=results_dir,
            start_time=time.time(),
            generation=current_gen,
            parent_id=parent_id,
            archive_insp_ids=archive_insp_ids,
            top_k_insp_ids=top_k_insp_ids,
            code_diff=code_diff,
            meta_patch_data=meta_patch_data,
            code_embedding=code_embedding,
            embed_cost=embed_cost,
            novelty_cost=novelty_cost,
        )
        self.running_jobs.append(running_job)

        if self.verbose:
            logger.info(
                f"Submitted job for generation {current_gen}, "
                f"queue size: {len(self.running_jobs)}"
            )

    def _check_completed_jobs(self) -> List[RunningJob]:
        """Check for completed jobs and return them."""
        completed = []
        still_running = []

        for job in self.running_jobs:
            is_running = self.scheduler.check_job_status(job)
            if not is_running:
                # Job completed
                if self.verbose:
                    logger.info(f"Job {job.job_id} completed!")
                completed.append(job)
            else:
                # Job still running
                still_running.append(job)

        self.running_jobs = still_running
        return completed

    def _process_completed_job(self, job: RunningJob):
        """Process a completed job and add results to database."""
        end_time = time.time()
        rtime = end_time - job.start_time

        # Get job results
        results = self.scheduler.get_job_results(job.job_id, job.results_dir)

        # Read the evaluated code
        try:
            evaluated_code = Path(job.exec_fname).read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not read code for job {job.job_id}. Error: {e}")
            evaluated_code = ""

        # Use pre-computed embedding and novelty costs
        code_embedding = job.code_embedding
        e_cost = job.embed_cost
        n_cost = job.novelty_cost
        if self.verbose:
            logger.debug(
                f"=> Using pre-computed embedding for job {job.job_id}, "
                f"embed cost: {e_cost:.4f}, novelty cost: {n_cost:.4f}"
            )

        correct_val = False
        metrics_val = {}
        stdout_log = ""
        stderr_log = ""
        if results:
            correct_val = results.get("correct", {}).get("correct", False)
            metrics_val = results.get("metrics", {})
            stdout_log = results.get("stdout_log", "")
            stderr_log = results.get("stderr_log", "")

        combined_score = metrics_val.get("combined_score", 0.0)
        public_metrics = metrics_val.get("public", {})
        private_metrics = metrics_val.get("private", {})
        text_feedback = metrics_val.get("text_feedback", "")

        # Add the program to the database
        db_program = Program(
            id=str(uuid.uuid4()),
            code=evaluated_code,
            language=self.evo_config.language,
            parent_id=job.parent_id,
            generation=job.generation,
            archive_inspiration_ids=job.archive_insp_ids,
            top_k_inspiration_ids=job.top_k_insp_ids,
            code_diff=job.code_diff,
            embedding=code_embedding,
            correct=correct_val,
            combined_score=combined_score,
            public_metrics=public_metrics,
            private_metrics=private_metrics,
            text_feedback=text_feedback,
            metadata={
                "compute_time": rtime,
                **(job.meta_patch_data or {}),
                "embed_cost": e_cost,
                "novelty_cost": n_cost,
                "stdout_log": stdout_log,
                "stderr_log": stderr_log,
            },
        )
        self.db.add(db_program, verbose=True)

        # Add the evaluated program to meta memory tracking
        self.meta_summarizer.add_evaluated_program(db_program)

        # Check if we should update meta memory after adding this program
        if self.meta_summarizer.should_update_meta(self.evo_config.meta_rec_interval):
            logger.info(
                f"Updating meta memory after processing "
                f"{len(self.meta_summarizer.evaluated_since_last_meta)} programs..."
            )
            best_program = self.db.get_best_program()
            updated_recs, meta_cost = self.meta_summarizer.update_meta_memory(
                best_program
            )
            if updated_recs:
                # Write meta output file using accumulated program count
                self.meta_summarizer.write_meta_output(str(self.results_dir))
                # Store meta cost for tracking
                if meta_cost > 0:
                    logger.info(
                        f"Meta recommendation generation cost: ${meta_cost:.4f}"
                    )
                    # Add meta cost to this program's metadata (the one that triggered the update)
                    if db_program.metadata is None:
                        db_program.metadata = {}
                    db_program.metadata["meta_cost"] = meta_cost
                    # Update the program in the database with the new metadata
                    import json

                    metadata_json = json.dumps(db_program.metadata)
                    self.db.cursor.execute(
                        "UPDATE programs SET metadata = ? WHERE id = ?",
                        (metadata_json, db_program.id),
                    )
                    self.db.conn.commit()

        if self.llm_selection is not None:
            if "model_name" not in db_program.metadata:
                logger.warning(
                    "No model_name found in program metadata, "
                    "unable to update model selection algorithm."
                )
            else:
                parent = (
                    self.db.get(db_program.parent_id) if db_program.parent_id else None
                )
                baseline = parent.combined_score if parent else None
                reward = db_program.combined_score if correct_val else None
                model_name = db_program.metadata["model_name"]
                result = self.llm_selection.update(
                    arm=model_name,
                    reward=reward,
                    baseline=baseline,
                )
                if result and self.verbose:
                    normalized_score, baseline = result

                    def fmt(x):
                        return f"{x:.4f}" if isinstance(x, (float, int)) else "None"

                    logger.debug(
                        f"==> UPDATED LLM SELECTION: model: "
                        f"{model_name.split('/')[-1][-25:]}..., "
                        f"score: {fmt(normalized_score)}, "
                        f"raw score: {fmt(reward)}, baseline: {fmt(baseline)}"
                    )
                    self.llm_selection.print_summary()

        self.db.save()
        self._update_best_solution()

        # Note: Meta summarization check is now done after completed generations
        # are updated in the main loop to ensure correct timing

        # Save meta memory state after each job completion
        self._save_meta_memory()

    def _update_best_solution(self):
        """Checks and updates the best program."""
        best_programs = self.db.get_top_programs(n=1, correct_only=True)
        if not best_programs:
            if self.verbose:
                logger.debug(
                    "No correct programs found yet, cannot determine best solution."
                )
            return

        best_program = best_programs[0]

        if best_program.id == self.best_program_id:
            return  # No change

        self.best_program_id = best_program.id

        # Multi-seed gen 0 uses gen_0_islandN dirs; check metadata for actual path
        source_dir = f"{self.results_dir}/{FOLDER_PREFIX}_{best_program.generation}"
        if best_program.metadata and best_program.metadata.get("source_dir"):
            candidate = best_program.metadata["source_dir"]
            if Path(candidate).exists():
                source_dir = candidate

        best_dir = Path(self.results_dir) / "best"

        if not Path(source_dir).exists():
            logger.warning(
                f"Source dir {source_dir} not found for best program "
                f"{best_program.id[:8]}..., skipping copy."
            )
            return

        if best_dir.exists():
            shutil.rmtree(best_dir)

        shutil.copytree(source_dir, best_dir)

        if self.verbose:
            logger.info(
                f"New best program found: gen {best_program.generation}, "
                f"id {best_program.id[:6]}... "
                f"Copied to {best_dir}"
            )

    def run_patch(
        self,
        parent_program: Program,
        archive_programs: List[Program],
        top_k_programs: List[Program],
        generation: int,
        novelty_attempt: int = 1,
        resample_attempt: int = 1,
        island_idx: Optional[int] = None,
    ) -> tuple[Optional[str], dict, int]:
        """Run patch generation for a specific generation."""
        max_patch_attempts = self.evo_config.max_patch_attempts
        if self.verbose:
            logger.info(
                f"Edit Cycle {generation} -> {generation + 1}, "
                f"Max Patch Attempts: {max_patch_attempts}"
            )
        # Get current meta recommendations
        meta_recs, _, _ = self.meta_summarizer.get_current()
        # Construct edit / code change message
        patch_sys, patch_msg, patch_type = self.prompt_sampler.sample(
            parent=parent_program,
            archive_inspirations=archive_programs,
            top_k_inspirations=top_k_programs,
            meta_recommendations=meta_recs,
            island_idx=island_idx,
        )

        if patch_type in ["full", "cross"]:
            apply_patch = apply_full_patch
        elif patch_type == "diff":
            apply_patch = apply_diff_patch
        elif patch_type == "paper":
            raise NotImplementedError("Paper edit not implemented.")
            # apply_patch = apply_paper_patch
        else:
            raise ValueError(f"Invalid patch type: {patch_type}")

        total_costs = 0
        msg_history = []
        llm_kwargs = self.llm.get_kwargs()
        if self.llm_selection is not None:
            model_name = llm_kwargs["model_name"]
            self.llm_selection.update_submitted(model_name)
        code_diff = None  # Initialize code_diff
        num_applied_attempt = 0  # Initialize num_applied_attempt
        error_attempt = (
            "Max attempts reached without successful patch."  # Default error
        )
        patch_name = None
        patch_description = None
        output_path_attempt = None
        patch_txt_attempt = None
        patch_path = None
        diff_summary = {}

        for patch_attempt in range(max_patch_attempts):
            response = self.llm.query(
                msg=patch_msg,
                system_msg=patch_sys,
                msg_history=msg_history,
                llm_kwargs=llm_kwargs,
            )
            # print(response.content)
            if response is None or response.content is None:
                if self.verbose:
                    logger.info(
                        f"  PATCH ATTEMPT {patch_attempt + 1}/{max_patch_attempts} FAILURE. "
                        f"Error: LLM response content was None."
                    )
                # Prepare for next attempt or exit
                error_attempt = "LLM response content was None."
                num_applied_attempt = 0
                patch_txt_attempt = None
                if patch_attempt < max_patch_attempts - 1:
                    patch_msg = (
                        "The previous attempt to get an edit was not "
                        "successful because the LLM response was empty. "
                        "Try again."
                    )
                    if response:
                        msg_history = response.new_msg_history
                    continue
                else:  # Last attempt
                    break

            total_costs += response.cost  # Acc. cost
            patch_name = extract_between(
                response.content,
                "<NAME>",
                "</NAME>",
                False,
            )
            patch_description = extract_between(
                response.content,
                "<DESCRIPTION>",
                "</DESCRIPTION>",
                False,
            )

            # Apply the code patch (diff/full rewrite)
            (
                _,
                num_applied_attempt,
                output_path_attempt,
                error_attempt,
                patch_txt_attempt,
                patch_path,
            ) = apply_patch(
                original_str=parent_program.code,
                patch_str=response.content,
                patch_dir=f"{self.results_dir}/{FOLDER_PREFIX}_{generation}",
                language=self.evo_config.language,
                verbose=False,
            )

            if error_attempt is None and num_applied_attempt > 0:
                if patch_path:  # Ensure patch_path is not None
                    diff_summary = summarize_diff(
                        str(patch_path)
                    )  # Convert Path to str
                if self.verbose:
                    logger.info(
                        f"  PATCH ATTEMPT {patch_attempt + 1}/{max_patch_attempts} SUCCESS. "
                        f"Output: {output_path_attempt}, "
                        f"Patches Applied: {num_applied_attempt}."
                    )

                code_diff = patch_txt_attempt
                break  # Break from patch attempts
            else:
                error_str = (
                    str(error_attempt) if error_attempt else "No changes applied."
                )
                patch_msg = (
                    "The previous edit was not successful."
                    + " This was the error message: \n\n"
                    + error_str
                    + "\n\n Try again."
                )
                if self.verbose:
                    logger.info(
                        f"  PATCH ATTEMPT {patch_attempt + 1}/{max_patch_attempts} FAILURE. "
                        f"Error: '{error_str}', "
                        f"Patches Applied: {num_applied_attempt}."
                    )
                msg_history = response.new_msg_history
                code_diff = None
                if patch_attempt == max_patch_attempts - 1:  # Last attempt failed
                    # error_attempt is already set from apply_patch or default
                    pass

        # Only consider the diff summary for the original source file
        original_filename = f"original.{self.lang_ext}"
        if original_filename in diff_summary:
            diff_summary = diff_summary[original_filename]

        meta_edit_data = {
            "patch_type": patch_type,
            "api_costs": total_costs,
            "num_applied": num_applied_attempt,
            "patch_name": patch_name,
            "patch_description": patch_description,
            "error_attempt": error_attempt,
            "novelty_attempt": novelty_attempt,
            "resample_attempt": resample_attempt,
            "patch_attempt": patch_attempt + 1,
            **llm_kwargs,
            "llm_result": response.to_dict() if response else None,
            "diff_summary": diff_summary,
        }
        if self.verbose and num_applied_attempt > 0:
            self._print_metadata_table(meta_edit_data, generation)
        # Delete generation from meta_edit_data
        return code_diff, meta_edit_data, num_applied_attempt

    def get_code_embedding(self, exec_fname: str) -> tuple[List[float], float]:
        """Get the embedding of the code."""
        # Read the evaluated code
        try:
            evaluated_code = Path(exec_fname).read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not read code for job {exec_fname}. Error: {e}")
            evaluated_code = ""
        if evaluated_code != "":
            # Get the embedding of the initial program
            try:
                if self.embedding is not None:
                    redacted_code = redact_immutable(evaluated_code, no_state=True)
                    MAX_EMBED_CHARS = 28000
                    if len(redacted_code) > MAX_EMBED_CHARS:
                        redacted_code = redacted_code[:MAX_EMBED_CHARS]
                    if self.verbose:
                        logger.debug(
                            "=> EMBED: Code length - "
                            f"Original: {len(evaluated_code)} - "
                            f"Redacted: {len(redacted_code)}"
                        )

                    embedding_result, e_cost = self.embedding.get_embedding(
                        redacted_code
                    )
                else:
                    if self.verbose:
                        logger.debug("=> EMBED: No embedding model configured.")
                    embedding_result = []
                    e_cost = 0.0
                code_embedding = cast(List[float], embedding_result)
            except Exception as e:
                logger.warning(f"Could not embed code for job {exec_fname}. Error: {e}")
                code_embedding = []
                e_cost = 0.0
        else:
            code_embedding = []
            e_cost = 0.0
        return code_embedding, e_cost

    def _print_metadata_table(self, meta_data: dict, generation: int):
        """Display metadata in a formatted rich table."""
        # Create title with generation and attempt information
        title_parts = ["[bold magenta]Patch Metadata"]

        # Add generation if present
        if generation is not None:
            title_parts.append(
                f" - Gen {generation}/{self.evo_config.num_generations} - Novelty: {meta_data['novelty_attempt']}/{self.evo_config.max_novelty_attempts} - Resample: {meta_data['resample_attempt']}/{self.evo_config.max_patch_resamples} - Patch: {meta_data['patch_attempt']}/{self.evo_config.max_patch_attempts}"
            )

        # Add attempt information if present
        if all(
            key in meta_data
            for key in [
                "novelty_attempt",
                "resample_attempt",
                "patch_attempt",
                "generation",
            ]
        ):
            title_parts.append(
                f" (Novelty: {meta_data['novelty_attempt']}, "
                f"Resample: {meta_data['resample_attempt']}, "
                f"Patch: {meta_data['patch_attempt']})"
            )

        title_parts.append("[/bold magenta]")
        table = Table(
            title="".join(title_parts),
            show_header=True,
            header_style="bold cyan",
            border_style="magenta",
            box=rich.box.ROUNDED,
            width=120,  # Match display.py table width
        )
        table.add_column("Field", style="cyan bold", no_wrap=True, width=25)
        table.add_column("Value", style="green", overflow="fold", width=90)

        # Define display order and formatting for specific fields
        display_order = [
            "patch_type",
            "patch_name",
            "patch_description",
            "num_applied",
            "api_costs",
            "error_attempt",
        ]

        # Add ordered fields first
        for field_name in display_order:
            if field_name in meta_data:
                value = meta_data[field_name]
                if value is None:
                    formatted_value = "[dim]None[/dim]"
                elif field_name == "api_costs":
                    formatted_value = f"${value:.4f}"
                elif field_name == "error_attempt" and value is None:
                    formatted_value = "[green]Success[/green]"
                elif field_name == "error_attempt":
                    formatted_value = (
                        f"[red]{str(value)[:100]}...[/red]"
                        if len(str(value)) > 100
                        else f"[red]{value}[/red]"
                    )
                else:
                    formatted_value = str(value)

                table.add_row(field_name, formatted_value)

        # Add remaining fields (excluding llm_result, diff_summary, and header info)
        skip_fields = set(
            display_order
            + [
                "llm_result",
                "diff_summary",
                "generation",
                "novelty_attempt",
                "resample_attempt",
                "patch_attempt",
            ]
        )
        for field_key, field_value in meta_data.items():
            if field_key not in skip_fields:
                if field_value is None:
                    formatted_value = "[dim]None[/dim]"
                else:
                    formatted_value = (
                        str(field_value)[:100] + "..."
                        if len(str(field_value)) > 100
                        else str(field_value)
                    )
                table.add_row(field_key, formatted_value)

        # Add diff summary if available
        if "diff_summary" in meta_data and meta_data["diff_summary"]:
            diff_summary = meta_data["diff_summary"]
            if isinstance(diff_summary, dict):
                summary_text = ""
                for k, v in diff_summary.items():
                    summary_text += f"{k}: {v}; "
                table.add_row("diff_summary", summary_text.strip())
            else:
                table.add_row("diff_summary", str(diff_summary)[:200])

        self.console.print(table)

    def _save_meta_memory(self) -> None:
        """Save the meta memory state to disk."""
        meta_memory_path = Path(self.results_dir) / "meta_memory.json"
        self.meta_summarizer.save_meta_state(str(meta_memory_path))

    def _restore_meta_memory(self) -> None:
        """Restore the meta memory state from disk."""
        meta_memory_path = Path(self.results_dir) / "meta_memory.json"

        if self.verbose:
            logger.info(f"Attempting to restore meta memory from: {meta_memory_path}")

        success = self.meta_summarizer.load_meta_state(str(meta_memory_path))
        if success:
            logger.info("Successfully restored meta memory state")
        else:
            if meta_memory_path.exists():
                logger.warning(
                    f"Meta memory file exists but failed to load: {meta_memory_path}"
                )
            else:
                logger.info("No previous meta memory state found - starting fresh")
