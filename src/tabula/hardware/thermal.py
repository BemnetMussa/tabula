"""
Tabula AI — Hardware Thermal Safety Guard.
Monitors CPU package and GPU junction temperatures in real time via psutil / hwmon sensors.
Automatically throttles and pauses workload if temps climb near fan ramp thresholds,
and executes emergency shutdown if temps approach dangerous limits.
"""

import sys
import time
import psutil


class ThermalGuard:
    def __init__(
        self,
        warm_threshold: float = 72.0,       # At this temp, pause and let fans cool it down
        emergency_threshold: float = 80.0,  # At this temp, cleanly halt to protect hardware
        target_cool_temp: float = 62.0,     # Wait until temp falls below this before resuming
        cooldown_sleep_sec: float = 2.0,    # Regular breather between games
    ):
        self.warm_threshold = warm_threshold
        self.emergency_threshold = emergency_threshold
        self.target_cool_temp = target_cool_temp
        self.cooldown_sleep_sec = cooldown_sleep_sec

    def read_temperatures(self) -> tuple[float, float]:
        """Returns (cpu_temp, gpu_temp) in Celsius."""
        cpu = 0.0
        gpu = 0.0
        try:
            temps = psutil.sensors_temperatures()
            # 1. CPU temp: look for coretemp or dell_ddv CPU
            if "coretemp" in temps:
                cpu = max([t.current for t in temps["coretemp"] if t.current is not None] or [0.0])
            elif "dell_ddv" in temps:
                for entry in temps["dell_ddv"]:
                    if entry.label == "CPU" and entry.current is not None:
                        cpu = entry.current
                        break

            # 2. GPU temp: check dell_ddv Video sensor
            if "dell_ddv" in temps:
                for entry in temps["dell_ddv"]:
                    if entry.label == "Video" and entry.current is not None:
                        gpu = entry.current
                        break
        except Exception:
            pass

        return float(cpu), float(gpu)

    def get_badge(self) -> str:
        """Returns an ANSI-formatted status string for TUI header."""
        cpu, gpu = self.read_temperatures()
        max_t = max(cpu, gpu)
        if max_t >= self.emergency_threshold:
            status = "\033[48;2;220;38;38m\033[1;38;2;255;255;255m DANGER \033[0m"
        elif max_t >= self.warm_threshold:
            status = "\033[48;2;217;119;6m\033[1;38;2;255;255;255m WARM \033[0m"
        else:
            status = "\033[48;2;22;163;74m\033[1;38;2;255;255;255m COOL \033[0m"
        return f"{status} \033[2mCPU:\033[0m{int(cpu)}°C \033[2mGPU:\033[0m{int(gpu)}°C"

    def rest(self, seconds: float = None):
        """Standard breather to let heat dissipation keep up with compute."""
        dur = seconds if seconds is not None else self.cooldown_sleep_sec
        if dur > 0:
            time.sleep(dur)

    def check_and_cool(self, live=None) -> bool:
        """
        Checks temperatures.
        - If emergency threshold exceeded: raises SystemExit or halts.
        - If warm threshold exceeded: loops and sleeps until cooled down.
        """
        cpu, gpu = self.read_temperatures()
        max_t = max(cpu, gpu)

        # 1. EMERGENCY CUTOFF: Hardware safety shutoff before thermal trip
        if max_t >= self.emergency_threshold:
            msg = f"\n\n🚨 [THERMAL GUARD EMERGENCY] Temp reached {max_t:.1f}°C (CPU: {cpu:.1f}°C, GPU: {gpu:.1f}°C)!\n"
            msg += f"Exceeded safety threshold of {self.emergency_threshold}°C.\n"
            msg += "Airflow may be blocked (e.g. resting on lap/cloth). Halting pipeline immediately to protect hardware.\n"
            sys.stderr.write(msg)
            sys.stderr.flush()
            raise SystemExit(1)

        # 2. SOFT GUARD: Cooldown breather
        if max_t >= self.warm_threshold:
            sys.stdout.write(f"\n[Thermal Guard] Temps reached {max_t:.1f}°C (CPU: {cpu:.1f}°C, GPU: {gpu:.1f}°C).\n")
            sys.stdout.write(f"Pausing compute to allow fans to dissipate heat down to <{self.target_cool_temp}°C...\n")
            sys.stdout.flush()
            while True:
                time.sleep(3.0)
                cpu, gpu = self.read_temperatures()
                max_t = max(cpu, gpu)
                sys.stdout.write(f"[Thermal Guard] Cooling... CPU: {cpu:.1f}°C, GPU: {gpu:.1f}°C\r")
                sys.stdout.flush()
                if max_t <= self.target_cool_temp:
                    sys.stdout.write(f"\n[Thermal Guard] Temperature normalized to {max_t:.1f}°C. Resuming pipeline.\n")
                    sys.stdout.flush()
                    break

        return True
