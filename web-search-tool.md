Use this. Browser Use already has built-in `search`, `navigate`, `click`, `input`, `evaluate`, `extract`, and screenshot actions, so a **single Serena tool** can cover both web search and full browser interaction. Browser Use also documents persistent sessions with `Browser(keep_alive=True)`, follow-up tasks via `agent.add_new_task(...)`, and cleanup with `await browser.kill()`. Serena’s documented extension point is to subclass `serena.agent.Tool` and implement `apply(...)`. Browser Use supports **Ollama** through `ChatOllama`. ([Browser Use Cloud][1])

```python
# path: src/serena/tools/browser_use_stateful_tool.py
import asyncio
import json
import threading
from typing import Any

from browser_use import Agent, Browser, ChatOllama
from serena.agent import Tool


class _BrowserUseRuntime:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._lock = threading.RLock()

        self.browser: Browser | None = None
        self.agent: Agent | None = None
        self.model_name: str | None = None
        self.browser_config: dict[str, Any] | None = None
        self.last_result: dict[str, Any] | None = None

    def _ensure_loop(self) -> None:
        if self._loop is not None:
            return

        def _runner() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._ready.set()
            loop.run_forever()

        self._thread = threading.Thread(
            target=_runner,
            name="browser-use-stateful-loop",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()

    def _run(self, coro: Any) -> Any:
        self._ensure_loop()
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    async def _shutdown_browser(self) -> None:
        if self.browser is None:
            self.agent = None
            self.model_name = None
            self.browser_config = None
            return

        try:
            kill_fn = getattr(self.browser, "kill", None)
            close_fn = getattr(self.browser, "close", None)

            if callable(kill_fn):
                await kill_fn()
            elif callable(close_fn):
                await close_fn()
        finally:
            self.browser = None
            self.agent = None
            self.model_name = None
            self.browser_config = None

    async def _ensure_browser(
        self,
        *,
        headless: bool,
        use_system_chrome: bool,
        profile_directory: str | None,
    ) -> None:
        desired = {
            "headless": headless,
            "use_system_chrome": use_system_chrome,
            "profile_directory": profile_directory,
        }

        recreate = self.browser is None or self.browser_config != desired
        if not recreate:
            return

        await self._shutdown_browser()

        if use_system_chrome:
            self.browser = Browser.from_system_chrome(
                profile_directory=profile_directory,
                headless=headless,
                keep_alive=True,
            )
        else:
            self.browser = Browser(
                headless=headless,
                keep_alive=True,
            )

        start_fn = getattr(self.browser, "start", None)
        if callable(start_fn):
            await start_fn()

        self.browser_config = desired
        self.agent = None
        self.model_name = None

    async def _run_task(
        self,
        *,
        task: str,
        model: str,
        max_steps: int,
        headless: bool,
        use_system_chrome: bool,
        profile_directory: str | None,
        fresh_agent: bool,
    ) -> dict[str, Any]:
        await self._ensure_browser(
            headless=headless,
            use_system_chrome=use_system_chrome,
            profile_directory=profile_directory,
        )
        assert self.browser is not None

        if self.agent is None or self.model_name != model or fresh_agent:
            llm = ChatOllama(model=model)
            self.agent = Agent(
                task=task,
                llm=llm,
                browser_session=self.browser,
            )
            self.model_name = model
        else:
            self.agent.add_new_task(task)

        history = await self.agent.run(max_steps=max_steps)

        self.last_result = {
            "done": history.is_done(),
            "success": history.is_successful(),
            "final_result": history.final_result(),
            "urls": history.urls(),
            "actions": history.action_names(),
            "extracted_content": history.extracted_content(),
            "errors": [e for e in history.errors() if e],
            "screenshots": history.screenshot_paths(),
            "steps": history.number_of_steps(),
            "duration_seconds": history.total_duration_seconds(),
            "model": self.model_name,
            "browser": self.browser_config,
        }
        return self.last_result

    async def _status(self) -> dict[str, Any]:
        return {
            "session_active": self.browser is not None,
            "agent_active": self.agent is not None,
            "model": self.model_name,
            "browser": self.browser_config,
            "last_result": self.last_result,
        }

    async def _reset(self) -> dict[str, Any]:
        await self._shutdown_browser()
        self.last_result = None
        return {"ok": True, "reset": True}


_RUNTIME = _BrowserUseRuntime()


class BrowserUseStatefulTool(Tool):
    """
    Persistent Serena tool for Browser Use + Ollama.

    Actions:
    - run:   execute a browsing/search task in the current browser session
    - status: inspect whether the session is alive
    - reset: close the browser and clear agent/session state
    """

    def apply(
        self,
        action: str = "run",
        task: str | None = None,
        model: str = "llama3.1:8b",
        max_steps: int = 8,
        headless: bool = True,
        use_system_chrome: bool = False,
        profile_directory: str | None = None,
        fresh_agent: bool = False,
    ) -> str:
        with _RUNTIME._lock:
            action = action.strip().lower()

            if action == "run":
                if not task:
                    raise ValueError("task is required when action='run'")

                result = _RUNTIME._run(
                    _RUNTIME._run_task(
                        task=task,
                        model=model,
                        max_steps=max_steps,
                        headless=headless,
                        use_system_chrome=use_system_chrome,
                        profile_directory=profile_directory,
                        fresh_agent=fresh_agent,
                    )
                )
                return json.dumps(result, ensure_ascii=False, indent=2)

            if action == "status":
                result = _RUNTIME._run(_RUNTIME._status())
                return json.dumps(result, ensure_ascii=False, indent=2)

            if action == "reset":
                result = _RUNTIME._run(_RUNTIME._reset())
                return json.dumps(result, ensure_ascii=False, indent=2)

            raise ValueError("action must be one of: run, status, reset")
```

## What this does

This version keeps a **single browser session alive across Serena calls**. On the first `run`, it creates `Browser(keep_alive=True)` and an Ollama-backed `Agent`. On later `run` calls, it reuses the same browser and adds the new task with `agent.add_new_task(...)`, which is the Browser Use pattern for preserving cookies, local storage, page state, and agent memory across follow-up tasks. ([Browser Use Cloud][2])

## Use it like this

```text
browser_use_stateful_tool(
  action="run",
  task="search for Serena release notes and open the latest changelog entry",
  model="llama3.1:8b",
  headless=false
)
```

```text
browser_use_stateful_tool(
  action="run",
  task="extract the release highlights from the current page"
)
```

```text
browser_use_stateful_tool(action="status")
```

```text
browser_use_stateful_tool(action="reset")
```

## Notes that matter

Browser Use’s Ollama path is the simple one: `from browser_use import Agent, ChatOllama` and `llm = ChatOllama(model="...")`. For browser control, Browser Use documents `headless`, `keep_alive`, and `Browser.from_system_chrome(profile_directory=..., **kwargs)`. Using `from_system_chrome(...)` is the clean way to reuse your real Chrome profile for logged-in sites; the docs note that you may need to fully close Chrome first so the profile is not in use elsewhere. ([Browser Use Cloud][3])

Browser Use’s current install path is `uv add browser-use && uv sync`, with `uvx browser-use install` if Chromium is not already available. ([PyPI][4])

If your Serena fork does not auto-import new tool modules, add one import for this module during Serena startup. Serena’s README is clear on the contract: subclass `serena.agent.Tool`, implement `apply(...)`, and `SerenaAgent` can use the new tool. ([GitHub][5])

If you want the **same thing but with Ollama host/model knobs exposed explicitly** or a **search-only lightweight version**, say which one.

[1]: https://docs.browser-use.com/open-source/customize/tools/available "Available Tools - Browser Use"
[2]: https://docs.browser-use.com/open-source/examples/templates/follow-up-tasks "Follow up tasks - Browser Use"
[3]: https://docs.browser-use.com/open-source/supported-models "Supported Models - Browser Use"
[4]: https://pypi.org/project/browser-use/?utm_source=chatgpt.com "browser-use"
[5]: https://github.com/oraios/serena?utm_source=chatgpt.com "oraios/serena: A powerful coding agent toolkit providing ..."


Use the **stateful Ollama path** for this tool.

* **LLM:** `ChatOllama(model="...")`
* **Browser session:** `Browser(keep_alive=True)`
* **Follow-up calls:** `agent.add_new_task(...)`
* **Serena integration:** custom Serena tool via `serena.agent.Tool` with `apply(...)` ([Browser Use Cloud][1])

That is the documented combination that matches a **persistent Serena browsing tool**: Browser Use’s Ollama example uses `ChatOllama`, and its follow-up-task example uses `Browser(keep_alive=True)` plus `agent.add_new_task(...)` to preserve browser state across multiple tasks. ([Browser Use Cloud][1])

If you need **logged-in sites / real Chrome profile reuse**, swap the browser constructor to `Browser.from_system_chrome()` instead of plain `Browser(...)`. Browser Use documents that path specifically for preserving login sessions, cookies, and extensions. ([Browser Use Cloud][2])

So for the tool itself, keep:

```python
llm = ChatOllama(model="llama3.1:8b")
browser = Browser(keep_alive=True)
```

And only switch to:

```python
browser = Browser.from_system_chrome()
```

when the task needs your real browser session. ([Browser Use Cloud][1])

**Bottom line:** use the **stateful Ollama version**, not the lightweight search-only variant.

