import re
import statistics
import subprocess
import threading
import time

import numpy as np


VDD_IN_PATTERN = re.compile(r"\bVDD_IN\s+(\d+)mW(?:/(\d+)mW)?")
RAM_PATTERN = re.compile(r"\bRAM\s+(\d+)/(\d+)MB")
TEMPERATURE_PATTERN = re.compile(r"([A-Za-z0-9_]+)@(-?\d+(?:\.\d+)?)C")
GPU_PATTERN = re.compile(r"\bGR3D_FREQ\s+(\d+)%")
CLOCK_PAIR_PATTERN = re.compile(
    r"MinFreq=(\d+)\b.*?MaxFreq=(\d+)\b", re.IGNORECASE
)


def percentile(values, fraction):
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return float(ordered[index])


def parse_tegrastats_line(line):
    sample = {"raw": line.rstrip()}
    power = VDD_IN_PATTERN.search(line)
    if power:
        sample["vdd_in_mw"] = int(power.group(1))
        if power.group(2) is not None:
            sample["vdd_in_average_mw"] = int(power.group(2))
    ram = RAM_PATTERN.search(line)
    if ram:
        sample["ram_used_mb"] = int(ram.group(1))
        sample["ram_total_mb"] = int(ram.group(2))
    temperatures = {
        name: float(value) for name, value in TEMPERATURE_PATTERN.findall(line)
    }
    if temperatures:
        sample["temperatures_c"] = temperatures
    gpu = GPU_PATTERN.search(line)
    if gpu:
        sample["gpu_utilization_percent"] = int(gpu.group(1))
    return sample


def clocks_are_locked(output):
    domains = {"cpu": [], "gpu": [], "emc": []}
    for line in output.splitlines():
        match = CLOCK_PAIR_PATTERN.search(line)
        if match is None:
            continue
        normalized = line.strip().lower()
        if re.match(r"cpu\d+\s*:", normalized):
            domain = "cpu"
        elif re.search(r"\bgpu\b", normalized):
            domain = "gpu"
        elif re.search(r"\bemc\b", normalized):
            domain = "emc"
        else:
            continue
        domains[domain].append((int(match.group(1)), int(match.group(2))))
    return all(
        pairs and all(low == high for low, high in pairs)
        for pairs in domains.values()
    )


def summarize_power(samples, idle_count, mean_latency_ms):
    if idle_count <= 0 or idle_count >= len(samples):
        raise ValueError("Need nonempty idle and active tegrastats samples")
    idle = [sample["vdd_in_mw"] for sample in samples[:idle_count] if "vdd_in_mw" in sample]
    active_samples = samples[idle_count:]
    active = [sample["vdd_in_mw"] for sample in active_samples if "vdd_in_mw" in sample]
    if not idle or not active:
        raise RuntimeError("tegrastats did not report VDD_IN for idle and active windows")
    idle_mean = statistics.fmean(idle)
    active_mean = statistics.fmean(active)
    peak_power = max(active)
    ram_values = [sample["ram_used_mb"] for sample in active_samples if "ram_used_mb" in sample]
    temperatures = [
        value
        for sample in active_samples
        for value in sample.get("temperatures_c", {}).values()
    ]
    seconds_per_image = mean_latency_ms / 1000.0
    return {
        "idle_samples": len(idle),
        "active_samples": len(active),
        "idle_vdd_in_w": idle_mean / 1000.0,
        "active_vdd_in_mean_w": active_mean / 1000.0,
        "active_vdd_in_peak_w": peak_power / 1000.0,
        "dynamic_power_mean_w": max(0.0, active_mean - idle_mean) / 1000.0,
        "total_energy_j_per_image": active_mean * seconds_per_image / 1000.0,
        "dynamic_energy_j_per_image": max(0.0, active_mean - idle_mean)
        * seconds_per_image
        / 1000.0,
        "peak_ram_mb": max(ram_values) if ram_values else None,
        "peak_temperature_c": max(temperatures) if temperatures else None,
    }


class TegrastatsMonitor:
    def __init__(self, interval_ms=100):
        if interval_ms <= 0:
            raise ValueError("tegrastats interval must be positive")
        self.interval_ms = interval_ms
        self.samples = []
        self.process = None
        self.thread = None

    def _read(self):
        for line in self.process.stdout:
            sample = parse_tegrastats_line(line)
            sample["monotonic_time"] = time.monotonic()
            self.samples.append(sample)

    def start(self):
        self.process = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def stop(self):
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        self.thread.join(timeout=5)


def load_cudart():
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        try:
            from cuda import cudart
        except ImportError as error:
            raise RuntimeError(
                "Install cuda-python matching JetPack before Phase 30"
            ) from error
    return cudart


def cuda_result(result, operation):
    values = result if isinstance(result, tuple) else (result,)
    error = values[0]
    if int(error) != 0:
        raise RuntimeError(f"CUDA {operation} failed with error code {int(error)}")
    if len(values) == 1:
        return None
    if len(values) == 2:
        return values[1]
    return values[1:]


class StandaloneTensorRTRunner:
    """One-input/one-output TensorRT runner backed only by cuda-python."""

    def __init__(self, engine_path):
        try:
            import tensorrt as trt
        except ImportError as error:
            raise RuntimeError("TensorRT Python bindings are required") from error
        self.trt = trt
        self.cudart = load_cudart()
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Could not create TensorRT execution context")
        inputs = []
        outputs = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append(name)
            else:
                outputs.append(name)
        if inputs != ["rgb_images"] or outputs != ["seg_mask"]:
            raise RuntimeError(
                f"Expected rgb_images -> seg_mask engine, found {inputs} -> {outputs}"
            )
        self.input_name = inputs[0]
        self.output_name = outputs[0]
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        if self.input_shape != (1, 3, 512, 512):
            raise RuntimeError(f"Unexpected input shape: {self.input_shape}")
        if not self.context.set_input_shape(self.input_name, self.input_shape):
            raise RuntimeError("TensorRT rejected the static input shape")
        self.output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        if self.output_shape != (1, 512, 512):
            raise RuntimeError(f"Unexpected output shape: {self.output_shape}")
        self.input_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(self.input_name)))
        self.output_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(self.output_name)))
        if self.input_dtype != np.dtype(np.float32) or self.output_dtype != np.dtype(np.uint8):
            raise RuntimeError(
                f"Unexpected engine dtypes: {self.input_dtype} -> {self.output_dtype}"
            )
        self.host_output = np.empty(self.output_shape, dtype=self.output_dtype)
        self.input_nbytes = int(np.prod(self.input_shape)) * self.input_dtype.itemsize
        self.output_nbytes = self.host_output.nbytes
        self.stream = cuda_result(self.cudart.cudaStreamCreate(), "stream create")
        self.device_input = cuda_result(
            self.cudart.cudaMalloc(self.input_nbytes), "input allocation"
        )
        self.device_output = cuda_result(
            self.cudart.cudaMalloc(self.output_nbytes), "output allocation"
        )
        if not self.context.set_tensor_address(
            self.input_name, int(self.device_input)
        ) or not self.context.set_tensor_address(
            self.output_name, int(self.device_output)
        ):
            raise RuntimeError("TensorRT rejected an I/O device address")

    def infer(self, input_array):
        input_array = np.ascontiguousarray(input_array, dtype=self.input_dtype)
        if input_array.shape != self.input_shape:
            raise ValueError(f"Expected input {self.input_shape}, got {input_array.shape}")
        kind = self.cudart.cudaMemcpyKind
        cuda_result(
            self.cudart.cudaMemcpyAsync(
                self.device_input,
                input_array.ctypes.data,
                self.input_nbytes,
                kind.cudaMemcpyHostToDevice,
                self.stream,
            ),
            "H2D copy",
        )
        if not self.context.execute_async_v3(int(self.stream)):
            raise RuntimeError("TensorRT execute_async_v3 returned false")
        cuda_result(
            self.cudart.cudaMemcpyAsync(
                self.host_output.ctypes.data,
                self.device_output,
                self.output_nbytes,
                kind.cudaMemcpyDeviceToHost,
                self.stream,
            ),
            "D2H copy",
        )
        cuda_result(self.cudart.cudaStreamSynchronize(self.stream), "stream sync")
        return self.host_output

    def close(self):
        for pointer in (getattr(self, "device_input", None), getattr(self, "device_output", None)):
            if pointer is not None:
                cuda_result(self.cudart.cudaFree(pointer), "device free")
        if getattr(self, "stream", None) is not None:
            cuda_result(self.cudart.cudaStreamDestroy(self.stream), "stream destroy")
        self.device_input = None
        self.device_output = None
        self.stream = None
