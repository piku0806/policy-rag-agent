"""Ask the policy agent a question.

    python -m policyrag "How long do I have to submit an expense report?"
"""
from __future__ import annotations

import json
import sys

from policyrag import tracing
from policyrag.graph import ask


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    tracing.setup()
    result = ask(" ".join(sys.argv[1:]))
    print(result["answer"])
    print("\n" + json.dumps({k: v for k, v in result.items() if k != "answer"}, indent=2))


if __name__ == "__main__":
    main()
