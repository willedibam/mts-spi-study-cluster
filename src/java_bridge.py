from __future__ import annotations

import os
import shutil
from importlib import metadata as importlib_metadata
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
    support_jar_path = _ensure_support_jar()
    if support_jar_path and not os.environ.get("JPYPE_JAR"):
        os.environ["JPYPE_JAR"] = support_jar_path
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
    classpath = [jar]
    if support_jar_path:
        classpath.append(support_jar_path)
    jpype.startJVM(jvm_path, "-ea", classpath=classpath)


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


def _ensure_support_jar() -> str | None:
    """
    JPype's C extension expects org.jpype.jar to sit next to the compiled module.
    Some site layouts (e.g. separate purelib/lib64) install the jar elsewhere; copy
    it into the location JPype probes so startJVM succeeds.
    """
    jpype_dir = Path(jpype.__file__).resolve().parent
    candidates = [
        jpype_dir / "org.jpype.jar",
        jpype_dir.parent / "org.jpype.jar",
    ]
    try:
        import jpype._core as _jp_core  # type: ignore[attr-defined]

        core_base = Path(_jp_core.__file__).resolve().parent.parent
        candidates.append(core_base / "org.jpype.jar")
    except Exception:
        core_base = None
    source = _first_existing(candidates) or _jar_from_distribution()
    if source is None:
        raise RuntimeError(
            "JPype support jar (org.jpype.jar) is missing; reinstall JPype1 "
            "or ensure the jar is present in your virtual environment."
        )
    if core_base is None:
        return str(source)
    target = core_base / "org.jpype.jar"
    try:
        if target.exists():
            return str(target)
        if source.resolve() == target.resolve():
            return str(target)
    except OSError:
        pass
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return str(target)
    except OSError as exc:  # pragma: no cover
        raise RuntimeError(
            f"Failed to copy JPype support jar to {target}: {exc}. "
            "Copy org.jpype.jar there manually or reinstall JPype1."
        ) from exc


def _jar_from_distribution() -> Path | None:
    try:
        files = importlib_metadata.files("JPype1") or []
    except importlib_metadata.PackageNotFoundError:
        return None
    for entry in files:
        if entry.name == "org.jpype.jar":
            located = Path(entry.locate())
            if located.exists():
                return located
    return None


def _first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
