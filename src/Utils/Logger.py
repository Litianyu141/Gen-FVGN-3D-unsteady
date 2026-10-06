"""Run directory of a training run: log file, copy of the source, residual history and
checkpoints."""

import csv
import datetime as dt
import json
import logging
import os
import shutil
import sys

_SOURCE_PACKAGES = ("NNmodels", "FVsolver", "FVdomain", "Utils", "Post_process", "Pipeline", "entries", "Diagnostics")


class Logger:
    """Creates ``<cwd>/<head>/<name>/<date-time>/`` and writes into it."""

    def __init__(self, name, head="Logger", datetime=None, use_csv=False, params=None, copy_code=False,
                 seed=None, log_level="INFO"):
        self.head = head
        self.name = name
        self.params = params
        self.datetime = datetime if datetime else dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.saving_path = os.path.join(os.getcwd(), head, name, self.datetime)
        os.makedirs(self.saving_path, exist_ok=True)
        self._setup_logging(log_level)
        if copy_code:
            self._copy_source_code()
        self.residual_buffer = []
        self.residual_buffer_size = 1000
        self.residual_csv_path = os.path.join(self.saving_path, "residuals.csv")
        self.residual_headers = None
        self.residual_headers_written = False
        self.residual_rows_written = 0

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def _setup_logging(self, log_level):
        """Root logging to stdout and ``training.log``."""
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)
        root_logger.handlers.clear()
        formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S")
        log_file = os.path.join(self.saving_path, "training.log")
        for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="a", encoding="utf-8")):
            handler.setLevel(numeric_level)
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)
        for lib in ("matplotlib", "PIL", "h5py", "urllib3", "numba"):
            logging.getLogger(lib).setLevel(logging.WARNING)
        self.logger = logging.getLogger(f"{self.head}.{self.name}")
        self.logger.info(f"Saving path: {self.saving_path}")

    def _copy_source_code(self):
        """Copy ``src/`` into the run directory (``source/``)."""
        source_dir = os.path.dirname(os.path.dirname(__file__))
        target_dir = os.path.join(self.saving_path, "source")
        os.makedirs(target_dir, exist_ok=True)
        for item in os.listdir(source_dir):
            if item in _SOURCE_PACKAGES and os.path.isdir(os.path.join(source_dir, item)):
                shutil.copytree(os.path.join(source_dir, item), os.path.join(target_dir, item), dirs_exist_ok=True)

    def save_state(self, model, optimizer, scheduler, index="final"):
        """Write ``states/<index>.state`` and the command-line arguments beside it."""
        states_dir = os.path.join(self.saving_path, "states")
        os.makedirs(states_dir, exist_ok=True)
        with open(os.path.join(states_dir, "commandline_args.json"), "wt") as f:
            json.dump(vars(self.params), f, indent=4, ensure_ascii=False)
        state_path = os.path.join(states_dir, f"{index}.state")
        model.save_checkpoint(state_path, optimizer, scheduler)
        self.logger.info(f"Model state saved to {state_path}")
        return state_path

    def load_state(self, model, datetime, index, device=None):
        """Load ``<head>/<name>/<datetime>/states/<index>.state`` into ``model``."""
        state_path = os.path.join(self.head, self.name, datetime, "states", f"{index}.state")
        model.load_checkpoint(state_path, device=device)
        self.logger.info(f"Model state loaded from {state_path}")
        return datetime, index

    def log_residuals(self, **kwargs):
        """Append one row (columns sorted by name) to ``residuals.csv``."""
        if self.residual_headers is None:
            self.residual_headers = sorted(kwargs.keys())
        self.residual_buffer.append([kwargs.get(h, 0.0) for h in self.residual_headers])
        if len(self.residual_buffer) >= self.residual_buffer_size:
            self._flush_residual_buffer()

    def _flush_residual_buffer(self):
        with open(self.residual_csv_path, "a" if self.residual_headers_written else "w", newline="") as f:
            writer = csv.writer(f)
            if not self.residual_headers_written:
                writer.writerow(self.residual_headers)
                self.residual_headers_written = True
            writer.writerows(self.residual_buffer)
        self.residual_rows_written += len(self.residual_buffer)
        self.residual_buffer.clear()

    def finalize_residuals(self):
        if self.residual_buffer:
            self._flush_residual_buffer()


class RankLogger:
    """Logger of DDP ranks other than 0: their own log file, no checkpoints or residual CSV."""

    def __init__(self, saving_path, rank):
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S")
        log_file = os.path.join(saving_path, f"training_rank{rank}.log")
        for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="a", encoding="utf-8")):
            handler.setLevel(logging.INFO)
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)
        self.saving_path = saving_path
        self._logger = logging.getLogger(f"Logger.rank{rank}")

    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def save_state(self, *args, **kwargs):
        pass

    def log_residuals(self, **kwargs):
        pass

    def finalize_residuals(self):
        pass


def ddp_run_logger(params, rank, world_size, device, seed, head="logger"):
    """Rank 0 creates the run directory and broadcasts its path; the other ranks log into it."""
    import torch
    import torch.distributed as dist
    from Utils.get_param import get_hyperparam

    if rank == 0:
        logger = Logger(get_hyperparam(params), head=head, use_csv=True, params=params, copy_code=True, seed=seed)
        path = logger.saving_path
    if world_size > 1:
        if rank == 0:
            path_bytes = path.encode("utf-8")
            path_len = torch.tensor([len(path_bytes)], dtype=torch.long, device=device)
            path_tensor = torch.tensor(list(path_bytes), dtype=torch.uint8, device=device)
        else:
            path_len = torch.zeros(1, dtype=torch.long, device=device)
        dist.broadcast(path_len, src=0)
        if rank != 0:
            path_tensor = torch.zeros(path_len.item(), dtype=torch.uint8, device=device)
        dist.broadcast(path_tensor, src=0)
        path = bytes(path_tensor.cpu().tolist()).decode("utf-8")
    if rank != 0:
        logger = RankLogger(path, rank)
    return logger, path
