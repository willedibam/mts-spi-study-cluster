from __future__ import annotations

import os
from pathlib import Path

import jpype
import pyspi


def ensure_java_started() -> None:
    """
    Start the JVM for PySPI's info-theoretic SPIs if it is not already running.

    The search order for the JVM shared library is:
    1. Explicit PYSPI_JVM env var (absolute path to libjvm.so / jvm.dll).
    2. JAVA_HOME env var with standard locations (lib/server/libjvm.so, etc.).
    3. jpype.getDefaultJVMPath() fallback.
    """
    if jpype.isJVMStarted():
        return
    jvm_path = (
        _jvm_from_env("PYSPI_JVM")
        or _jvm_from_java_home()
        or _default_jvm()
    )
    if not jvm_path:
        raise RuntimeError(
            "Cannot locate JVM; set PYSPI_JVM or JAVA_HOME to a valid JDK. "
            "Example: export PYSPI_JVM=/usr/lib/jvm/jdk-21/lib/server/libjvm.so"
        )
    jar = _pyspi_jidt_jar()
    jpype.startJVM(jvm_path, "-ea", classpath=[jar])


def _jvm_from_env(var: str) -> str | None:
    val = os.environ.get(var)
    if val and Path(val).exists():
        return val
    return None


def _jvm_from_java_home() -> str | None:
    java_home = os.environ.get("JAVA_HOME")
    if not java_home:
        return None
    candidates = [
        Path(java_home) / "lib" / "server" / "libjvm.so",
        Path(java_home) / "jre" / "lib" / "server" / "libjvm.so",
        Path(java_home) / "bin" / "server" / "jvm.dll",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _default_jvm() -> str | None:
    try:
        return jpype.getDefaultJVMPath()
    except (RuntimeError, OSError):
        return None


def _pyspi_jidt_jar() -> str:
    pyspi_root = Path(pyspi.__file__).resolve().parent
    jar = pyspi_root / "lib" / "jidt" / "infodynamics.jar"
    if not jar.exists():
        raise FileNotFoundError(
            f"PySPI JIDT jar not found at {jar}. Check your pyspi installation."
        )
    return str(jar)


# Auto-start when the module is imported so downstream code can assume TE-ready
ensure_java_started()

