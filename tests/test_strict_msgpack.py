import os
import subprocess
import sys
from pathlib import Path


def test_strict_msgpack_is_enabled_before_langgraph_import():
    env = os.environ.copy()
    env.pop("LANGGRAPH_STRICT_MSGPACK", None)
    src = Path(__file__).parents[1] / "src"
    env["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(src), env.get("PYTHONPATH", ""))
        if value
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import rag_app.assistant\n"
                "from langgraph.checkpoint.serde import _msgpack\n"
                "print(_msgpack.STRICT_MSGPACK_ENABLED)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.strip() == "True"
