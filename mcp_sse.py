# -*- coding: utf-8 -*-
import asyncio
import os
import sys
import traceback
import logging
from typing import Optional

from browser_use import BrowserConfig
from browser_use.browser.context import BrowserContextConfig
from fastmcp import FastMCP
from mcp.types import TextContent

from src.agent.custom_agent import CustomAgent
from src.browser.custom_browser import CustomBrowser
from src.controller.custom_controller import CustomController
from src.utils import utils
from src.utils.agent_state import AgentState

# Configure logging for the entire module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global references for single "running agent" approach
_global_agent: Optional[CustomAgent] = None
_global_browser: Optional[CustomBrowser] = None
from browser_use.browser.context import BrowserContext

_global_browser_context: Optional[BrowserContext] = None
_global_agent_state: AgentState = AgentState()

app = FastMCP("mcp_browser_use")


async def _cleanup_browser_resources() -> None:
    global _global_browser, _global_agent_state, _global_browser_context, _global_agent

    try:
        if _global_agent_state:
            try:
                await _global_agent_state.request_stop()
            except Exception as stop_error:
                logger.warning("Error stopping agent state: %s", stop_error)

        if _global_browser_context:
            try:
                await _global_browser_context.close()
            except Exception as context_error:
                logger.warning("Error closing browser context: %s", context_error)

        if _global_browser:
            try:
                await _global_browser.close()
            except Exception as browser_error:
                logger.warning("Error closing browser: %s", browser_error)

    except Exception as e:
        logger.error("Unexpected error during browser cleanup: %s", e)
    finally:
        _global_browser = None
        _global_browser_context = None
        _global_agent_state = AgentState()
        _global_agent = None


@app.tool()
async def run_browser_agent(task: str, add_infos: str = "") -> str:
    global _global_agent, _global_browser, _global_browser_context, _global_agent_state

    try:
        # Clear any previous agent stop signals
        _global_agent_state.clear_stop()

        # Utility functions to safely read environment variables
        def safe_float(env_var: str, default: float) -> float:
            try:
                return float(os.getenv(env_var, str(default)))
            except ValueError:
                logger.warning(f"Invalid float for {env_var}, using default={default}")
                return default

        def safe_int(env_var: str, default: int) -> int:
            try:
                return int(os.getenv(env_var, str(default)))
            except ValueError:
                logger.warning(f"Invalid int for {env_var}, using default={default}")
                return default

        def safe_bool(env_var: str, default: bool) -> bool:
            value = os.getenv(env_var, str(default)).lower()
            return value in ["true", "1", "yes"]

        # Read key configurations from environment variables with defaults
        model_provider = os.getenv("MCP_MODEL_PROVIDER", "anthropic")
        model_name = os.getenv("MCP_MODEL_NAME", "claude-3-5-sonnet-20241022")
        temperature = safe_float("MCP_TEMPERATURE", 0.3)
        max_steps = safe_int("MCP_MAX_STEPS", 30)
        use_vision = safe_bool("MCP_USE_VISION", True)
        max_actions_per_step = safe_int("MCP_MAX_ACTIONS_PER_STEP", 5)
        tool_call_in_content = safe_bool("MCP_TOOL_CALL_IN_CONTENT", True)
        chrome_path = os.getenv("CHROME_PATH", None)
        anonymized_telemetry = safe_bool("ANONYMIZED_TELEMETRY", False)
        logging_level = os.getenv("BROWSER_USE_LOGGING_LEVEL", "info").lower()

        # Prepare logger based on logging level
        if logging_level == "debug":
            logger.setLevel(logging.DEBUG)
        elif logging_level == "info":
            logger.setLevel(logging.INFO)

        # Prepare the LLM
        llm = utils.get_llm_model(
            provider=model_provider, model_name=model_name, temperature=temperature
        )

        # Create or reuse the global browser instance
        if not _global_browser:
            _global_browser = CustomBrowser(
                config=BrowserConfig(
                    headless=False,
                    disable_security=False,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=[],
                    wss_url=None,
                    proxy=None,
                )
            )

        # Create or reuse the global browser context
        if not _global_browser_context:
            _global_browser_context = await _global_browser.new_context(
                config=BrowserContextConfig(
                    trace_path=None, save_recording_path=None, no_viewport=False
                )
            )

        # Create controller and agent
        controller = CustomController()
        _global_agent = CustomAgent(
            task=task,
            add_infos=add_infos,
            use_vision=use_vision,
            llm=llm,
            browser=_global_browser,
            browser_context=_global_browser_context,
            controller=controller,
            max_actions_per_step=max_actions_per_step,
            tool_call_in_content=tool_call_in_content,
            agent_state=_global_agent_state,
        )

        # Run agent
        history = await _global_agent.run(max_steps=max_steps)

        # Extract final result from the agent's history
        final_result = history.final_result() or f"No final result. Possibly incomplete. {history}"

        return final_result

    except Exception as e:
        logger.error("run-browser-agent error: %s", str(e))
        raise ValueError(f"run-browser-agent error: {e}\n{traceback.format_exc()}")

    finally:
        await _cleanup_browser_resources()


def main() -> None:
    try:
        app.run(transport='sse')
    except Exception as e:
        logger.error("Error running MCP server: %s\n%s", e, traceback.format_exc())
    finally:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_cleanup_browser_resources())
            loop.close()
        except Exception as cleanup_error:
            logger.error("Cleanup error: %s", cleanup_error)


if __name__ == "__main__":
    main()
