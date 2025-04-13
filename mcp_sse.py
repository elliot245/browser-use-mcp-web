# -*- coding: utf-8 -*-
import asyncio
import os
import sys
import traceback
import logging
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from browser_use import BrowserConfig
from browser_use.browser.context import BrowserContextConfig, BrowserContextWindowSize
from fastmcp import FastMCP
from mcp.types import TextContent

from src.agent.custom_agent import CustomAgent
from src.browser.custom_browser import CustomBrowser
from src.controller.custom_controller import CustomController
from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt
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
                _global_agent_state.request_stop()
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
        chrome_cdp = os.getenv("CHROME_CDP", "")
        headless = safe_bool("MCP_HEADLESS", False)
        disable_security = safe_bool("MCP_DISABLE_SECURITY", True)
        window_w = safe_int("MCP_WINDOW_WIDTH", 1280)
        window_h = safe_int("MCP_WINDOW_HEIGHT", 1100)
        max_input_tokens = safe_int("MCP_MAX_INPUT_TOKENS", 128000)
        use_own_browser = safe_bool("MCP_USE_OWN_BROWSER", False)
        tool_calling_method = os.getenv("MCP_TOOL_CALLING_METHOD", "auto")
        
        # Browser configuration setup
        extra_chromium_args = ["--accept_downloads=True", f"--window-size={window_w},{window_h}"]
        
        cdp_url = chrome_cdp
        if use_own_browser:
            cdp_url = os.getenv("CHROME_CDP", chrome_cdp)
            chrome_user_data = os.getenv("CHROME_USER_DATA", None)
            if chrome_user_data:
                extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]

        # Prepare the LLM
        llm = utils.get_llm_model(
            provider=model_provider, 
            model_name=model_name, 
            temperature=temperature,
            num_ctx=max_input_tokens
        )

        # Create or reuse the global browser instance
        if (_global_browser is None) or (cdp_url and cdp_url != ""):
            _global_browser = CustomBrowser(
                config=BrowserConfig(
                    headless=headless,
                    disable_security=disable_security,
                    cdp_url=cdp_url,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                )
            )

        # Create or reuse the global browser context
        if (_global_browser_context is None) or (cdp_url and cdp_url != ""):
            _global_browser_context = await _global_browser.new_context(
                config=BrowserContextConfig(
                    trace_path=None, 
                    save_recording_path=None,
                    save_downloads_path="./tmp/downloads",
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(
                        width=window_w, 
                        height=window_h
                    ),
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
            system_prompt_class=CustomSystemPrompt,
            agent_prompt_class=CustomAgentMessagePrompt,
            max_actions_per_step=max_actions_per_step,
            tool_calling_method=tool_calling_method,
            max_input_tokens=max_input_tokens,
            generate_gif=False
        )

        # Run agent
        history = await _global_agent.run(max_steps=max_steps)

        # Extract final result from the agent's history
        final_result = history.final_result() or f"No final result. Possibly incomplete. {history}"

        return final_result

    except Exception as e:
        logger.error("run-browser-agent error: %s", str(e))
        traceback.print_exc()
        raise ValueError(f"run-browser-agent error: {e}\n{traceback.format_exc()}")

    finally:
        # If we want to keep browser open between runs, don't clean up
        keep_browser_open = safe_bool("MCP_KEEP_BROWSER_OPEN", False)
        if not keep_browser_open:
            await _cleanup_browser_resources()


def main() -> None:
    try:
        transport_type = os.getenv("MCP_TRANSPORT", "stdio")
        app.run(transport=transport_type)
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
