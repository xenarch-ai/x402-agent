# Packaging x402-agent as its own PyPI release

This package currently ships *inside* `xenarch-sdks/python` so it co-evolves with `XenarchPayer` during the refactor phase. Phase 1 of the upstream plan promotes it to a dedicated repo (`xenarch-ai/x402-agent`) and publishes it to PyPI.

When you run the split, this directory becomes the root of the new repo. Use the manifest below as the initial `pyproject.toml`.

## Target `pyproject.toml` (v0.1.0)

```toml
[project]
name = "x402-agent"
version = "0.1.0"
description = "Framework-agnostic payer for the x402 HTTP payment protocol. Works with LangChain, CrewAI, AutoGen, LangGraph, or any other agent framework."
readme = "README.md"
requires-python = ">=3.12"
license = {text = "MIT"}
authors = [
    {name = "Xenarch", email = "hello@xenarch.dev"},
]
keywords = [
    "x402", "http-402", "ai-agents", "langchain", "crewai",
    "autogen", "langgraph", "micropayments", "usdc", "base",
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Internet :: WWW/HTTP",
    "Topic :: Software Development :: Libraries :: Python Modules",
]
dependencies = [
    "httpx>=0.27",
    "pydantic>=2.0",
    "x402>=2.8,<3",
    "eth-account>=0.11",
]

[project.optional-dependencies]
pay-json = ["pay.json>=1.1,<2"]

[project.urls]
Homepage = "https://github.com/xenarch-ai/x402-agent"
Repository = "https://github.com/xenarch-ai/x402-agent"
Issues = "https://github.com/xenarch-ai/x402-agent/issues"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["x402_agent"]

[tool.pytest.ini_options]
asyncio_mode = "auto"

[tool.mypy]
files = ["x402_agent"]
strict = true

[[tool.mypy.overrides]]
module = ["x402.*", "pay_json.*"]
ignore_missing_imports = true
```

## Split procedure

1. `gh repo create xenarch-ai/x402-agent --public --description "Framework-agnostic x402 payer"`
2. `git subtree split -P python/x402_agent -b x402-agent-split`
3. `git clone git@github.com:xenarch-ai/x402-agent.git /tmp/x402-agent`
4. Cherry-pick the split branch into `/tmp/x402-agent`.
5. Move `PACKAGING.md` content into a new root `pyproject.toml`, delete `PACKAGING.md`.
6. Move `tests/x402_agent/` from `xenarch-sdks/python/` into `/tmp/x402-agent/tests/`.
7. `uv build && uv publish`.

Once published, update `xenarch-sdks/python/pyproject.toml`:

```diff
 dependencies = [
     "httpx>=0.27",
     "pydantic>=2.0",
     "cryptography>=42",
+    "x402-agent>=0.1,<0.2",
 ]
```

Remove the vendored copy:

```bash
git rm -r python/x402_agent/
git rm -r python/tests/x402_agent/
```

And drop `x402_agent` from `[tool.hatch.build.targets.wheel].packages`.
