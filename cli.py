"""
Command-line interface for GROMACS Copilot
"""

import os
import sys
import argparse
import logging

from gromacs_copilot.core.md_agent import MDLLMAgent
from gromacs_copilot.utils.terminal import Colors, print_message
from gromacs_copilot.utils.logging_utils import setup_logging
from gromacs_copilot.core.enums import MessageType
from gromacs_copilot.config import DEFAULT_WORKSPACE, DEFAULT_MODEL, DEFAULT_OPENAI_URL


def parse_arguments():
    """
    Parse command-line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="GROMACS Copilot")
    parser.add_argument("--api-key", help="API key for LLM service")
    parser.add_argument("--url", 
                      help=(
                          "The url of the LLM service, "
                          "\ndeepseek: https://api.deepseek.com/chat/completions"
                          "\nopenai: https://api.openai.com/v1/chat/completions"
                      ), 
                      default=DEFAULT_OPENAI_URL)
    # 修改模型参数处理，确保能正确处理包含冒号的模型名称
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to use for LLM (e.g. gpt-4, qwen3:32b)")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE, help="Workspace directory")
    parser.add_argument("--prompt", help="Starting prompt for the LLM", type=str)
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("--log-file", default="md_agent.log", help="Log file path")
    parser.add_argument("--log-level", default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--timeout", type=int, default=60, help="超时时间（秒），用于API请求")
    parser.add_argument("--mode", default="copilot", choices=['copilot', 'agent'],
                        help="The copilot mode or agent mode, copilot will be more like a advisor."
                        )
    
    return parser.parse_args()


def main():
    """
    Main entry point for the CLI
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    setup_logging(args.log_file, level=log_level)
    
    # Disable colors if requested or if not in a terminal
    if args.no_color or not sys.stdout.isatty():
        Colors.disable_colors()
    
    # Display splash screen
    print_message("", style="divider")
    print_message("GROMACS Copilot", MessageType.TITLE, style="box")
    print_message("A molecular dynamics simulation assistant powered by AI, created by the ChatMol Team.", MessageType.INFO)
    print_message("", style="divider")
    
    try:
        # Check for API key
        if args.url == "https://api.openai.com/v1/chat/completions":
            api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        elif args.url == "https://api.deepseek.com/chat/completions":
            api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
        else:
            # For other URLs or local URLs, try to get the key but don't enforce it strictly for local
            api_key = args.api_key
            # 处理特殊情况，当api_key是'~'或空字符串时，对于本地URL设置为None
            if api_key in ['~', '']:
                api_key = None

        # Only enforce API key check if the URL is not pointing to localhost
        is_local_url = "localhost" in args.url or "127.0.0.1" in args.url
        if not api_key and not is_local_url:
            print_message(
                "API key not found. Please provide an API key using --api-key or set the "
                "appropriate environment variable (e.g., OPENAI_API_KEY, DEEPSEEK_API_KEY).", 
                MessageType.ERROR
            )
            sys.exit(1)
        
        # Create and run MD LLM agent
        print_message(f"Initializing with model: {args.model}", MessageType.INFO)
        print_message(f"Using workspace: {args.workspace}", MessageType.INFO)
        
        agent = MDLLMAgent(
            api_key=api_key, 
            model=args.model, 
            workspace=args.workspace, 
            url=args.url,
            mode=args.mode
        )
        
        # 设置自定义超时时间（如果指定）
        if hasattr(args, 'timeout') and args.timeout:
            agent.request_timeout = args.timeout
            logging.info(f"已设置API请求超时时间为 {args.timeout} 秒")
        agent.run(starting_prompt=args.prompt)
        
    except KeyboardInterrupt:
        print_message("\nExiting the MD agent. Thank you for using GROMACS Copilot!", 
                     MessageType.SUCCESS, style="box")
    except Exception as e:
        error_msg = str(e)
        logging.error(f"Error running MD LLM agent: {error_msg}")
        print_message(f"Error running MD LLM agent: {error_msg}", 
                     MessageType.ERROR, style="box")


if __name__ == "__main__":
    main()