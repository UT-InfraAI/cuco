import logging
import time
import asyncio
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, Tuple, Union, List
from concurrent.futures import ThreadPoolExecutor
from .local import submit as submit_local, monitor as monitor_local
from .local import ProcessWithLogging
from cuco.utils import parse_time_to_seconds

logger = logging.getLogger(__name__)


@dataclass
class JobConfig:
    """Base job configuration"""

    eval_program_path: Optional[str] = "evaluate.py"
    extra_cmd_args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        job_to_dict = asdict(self)
        return {k: v for k, v in job_to_dict.items() if v is not None}


@dataclass
class LocalJobConfig(JobConfig):
    """Configuration for local jobs"""

    time: Optional[str] = None
    conda_env: Optional[str] = None


class JobScheduler:
    def __init__(
        self,
        job_type: str,
        config: LocalJobConfig,
        verbose: bool = False,
        max_workers: int = 4,
    ):
        self.job_type = job_type
        self.config = config
        self.verbose = verbose
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        if self.job_type == "local":
            self.monitor = monitor_local
        else:
            raise ValueError(
                f"Unknown job type: {job_type}. Must be 'local'"
            )

    def _build_command(self, exec_fname_t: str, results_dir_t: str) -> List[str]:
        assert isinstance(self.config, LocalJobConfig)
        if self.config.conda_env:
            cmd = [
                "conda",
                "run",
                "-n",
                self.config.conda_env,
                "python",
                f"{self.config.eval_program_path}",
                "--program_path",
                f"{exec_fname_t}",
                "--results_dir",
                results_dir_t,
            ]
        else:
            cmd = [
                "python",
                f"{self.config.eval_program_path}",
                "--program_path",
                f"{exec_fname_t}",
                "--results_dir",
                results_dir_t,
            ]
        if self.config.extra_cmd_args:
            for k, v in self.config.extra_cmd_args.items():
                if isinstance(v, bool):
                    if v:
                        cmd.append(f"--{k}")
                else:
                    cmd.extend([f"--{k}", str(v)])
        return cmd

    def run(
        self, exec_fname_t: str, results_dir_t: str
    ) -> Tuple[Dict[str, Any], float]:
        cmd = self._build_command(exec_fname_t, results_dir_t)
        start_time = time.time()

        assert isinstance(self.config, LocalJobConfig)
        job_id = submit_local(results_dir_t, cmd, verbose=self.verbose)

        results = monitor_local(job_id, results_dir_t)
        end_time = time.time()
        rtime = end_time - start_time

        if results is None:
            results = {"correct": {"correct": False}, "metrics": {}}

        return results, rtime

    def submit_async(
        self, exec_fname_t: str, results_dir_t: str
    ) -> ProcessWithLogging:
        """Submit a job asynchronously and return the process."""
        cmd = self._build_command(exec_fname_t, results_dir_t)
        assert isinstance(self.config, LocalJobConfig)
        return submit_local(results_dir_t, cmd, verbose=self.verbose)

    def check_job_status(self, job) -> bool:
        """Check if job is running. Returns True if running, False if done."""
        if isinstance(job.job_id, ProcessWithLogging):
            if (
                isinstance(self.config, LocalJobConfig)
                and self.config.time
                and job.start_time
            ):
                timeout = parse_time_to_seconds(self.config.time)
                if time.time() - job.start_time > timeout:
                    if self.verbose:
                        logger.warning(
                            f"Process {job.job_id.pid} exceeded "
                            f"timeout of {self.config.time}. Killing. "
                            f"=> Gen. {job.generation}"
                        )
                    job.job_id.kill()
                    return False

            try:
                return job.job_id.poll() is None
            except Exception as e:
                logger.warning(f"poll() failed for PID {job.job_id.pid}: {e}")
                try:
                    import psutil
                    return psutil.pid_exists(job.job_id.pid)
                except ImportError:
                    try:
                        import os
                        os.kill(job.job_id.pid, 0)
                        return True
                    except (OSError, ProcessLookupError):
                        return False
                except Exception as e2:
                    logger.warning(
                        f"All status check methods failed for PID {job.job_id.pid}: {e2}"
                    )
                    return False
        return False

    def get_job_results(
        self, job_id: ProcessWithLogging, results_dir: str
    ) -> Optional[Dict[str, Any]]:
        """Get results from a completed job."""
        if isinstance(job_id, ProcessWithLogging):
            job_id.wait()
            return monitor_local(
                job_id,
                results_dir,
                verbose=self.verbose,
                timeout=self.config.time,
            )
        return None

    async def submit_async_nonblocking(
        self, exec_fname_t: str, results_dir_t: str
    ) -> ProcessWithLogging:
        """Submit a job asynchronously without blocking the event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self.submit_async, exec_fname_t, results_dir_t
        )

    async def check_job_status_async(self, job) -> bool:
        """Async version of job status checking."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.check_job_status, job)

    async def get_job_results_async(
        self, job_id: ProcessWithLogging, results_dir: str
    ) -> Optional[Dict[str, Any]]:
        """Async version of getting job results."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self.get_job_results, job_id, results_dir
        )

    async def batch_check_status_async(self, jobs: List) -> List[bool]:
        """Check status of multiple jobs concurrently."""
        tasks = [self.check_job_status_async(job) for job in jobs]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def cancel_job_async(self, job_id: ProcessWithLogging) -> bool:
        """Cancel a running job asynchronously."""
        loop = asyncio.get_event_loop()

        def cancel_job():
            try:
                if isinstance(job_id, ProcessWithLogging):
                    job_id.kill()
                    return True
                return False
            except Exception as e:
                logger.error(f"Error cancelling job {job_id}: {e}")
                return False

        return await loop.run_in_executor(self.executor, cancel_job)

    def shutdown(self):
        """Shutdown the thread pool executor."""
        self.executor.shutdown(wait=True)
