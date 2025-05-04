"""
Main MD Agent class for GROMACS Copilot
"""

import os
import json
import logging
import requests
from typing import List, Dict, Any, Optional, Union

from gromacs_copilot.protocols.protein import ProteinProtocol
from gromacs_copilot.protocols.protein_ligand import ProteinLigandProtocol
from gromacs_copilot.protocols.mmpbsa import MMPBSAProtocol
from gromacs_copilot.protocols.analysis import AnalysisProtocol

from gromacs_copilot.utils.terminal import print_message, prompt_user
from gromacs_copilot.core.enums import MessageType, SimulationStage
from gromacs_copilot.config import SYSTEM_MESSAGE_ADVISOR, SYSTEM_MESSAGE_AGENT


class MDLLMAgent:
    """LLM-based agent for running molecular dynamics simulations with GROMACS"""
    
    def __init__(self, api_key: str = None, model: str = "qwen3:32b", 
                workspace: str = "./md_workspace", 
                url: str = "http://localhost:11434/v1/chat/completions", mode: str = "copilot", gmx_bin: str = "gmx"):
        """
        Initialize the MD LLM agent
        
        Args:
            api_key: API key for LLM service
            model: Model to use for LLM
            workspace: Directory to use as the working directory
            url: URL of the LLM API endpoint
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        # 处理特殊情况，当api_key是'~'或空字符串时设置为None
        if self.api_key in ['~', '']:
            self.api_key = None
            
        self.url = url
        self.original_url = url  # Store original URL for fallback
        self.original_model = model  # Store original model for fallback
        
        # 检查是否是本地URL
        is_local_url = "localhost" in url or "127.0.0.1" in url
        
        # For local ollama model, api_key is not required
        if not is_local_url and not self.api_key:
            raise ValueError("API key is required for non-local models. Provide as parameter or set OPENAI_API_KEY environment variable")
            
        # Set timeout and retry parameters for API requests
        self.request_timeout = 60  # 增加超时时间以适应较慢的网络连接
        self.max_retries = 3  # 最大重试次数
        self.retry_delay = 2  # 重试间隔（秒）
        
        self.model = model
        self.conversation_history = []
        self.workspace = workspace
        self.gmx_bin = gmx_bin
        
        # Initialize protocol (will be set to protein or protein-ligand as needed)
        self.protocol = ProteinProtocol(workspace, self.gmx_bin)
        self.mode = mode
        
        logging.info(f"MD LLM Agent initialized with model: {model}")
        
    def _handle_api_error(self, error):
        """
        Handle API connection errors by falling back to local ollama model
        
        Args:
            error: Exception object from failed API request
        """
        if isinstance(error, (requests.exceptions.ConnectTimeout, 
                            requests.exceptions.ConnectionError)):
            # 提供更详细的错误信息
            error_message = str(error)
            logging.warning(f"API 连接失败: {error_message}")
            
            # 检查是否已经是本地Ollama模型
            if "localhost" in self.url or "127.0.0.1" in self.url:
                # 如果已经是本地Ollama，提供更具体的错误处理建议
                logging.error("无法连接到本地Ollama服务。请确保Ollama服务正在运行。")
                logging.info("可以通过运行 'ollama serve' 命令启动Ollama服务。")
                logging.info("如果Ollama服务已在运行，可能是由于网络延迟或服务负载过高导致超时。")
                # 仍然抛出异常，因为没有备用选项
                raise error
            else:
                # 如果不是本地Ollama，尝试回退到本地Ollama
                logging.warning(f"尝试回退到本地Ollama模型")
                self.url = "http://localhost:11434/v1/chat/completions"
                self.model = "qwen3:32b"
                logging.info(f"已切换到本地Ollama模型: {self.url}")
        else:
            # 其他类型的错误
            logging.error(f"API错误: {str(error)}")
            raise error

    def switch_to_mmpbsa_protocol(self) -> Dict[str, Any]:
        """
        Switch to MM-PBSA protocol
        
        Returns:
            Dictionary with result information
        """
        try:
            # Create new MM-PBSA protocol
            old_protocol = self.protocol
            self.protocol = MMPBSAProtocol(self.workspace)
            
            # Copy relevant state from the old protocol if possible
            if hasattr(old_protocol, 'topology_file'):
                self.protocol.topology_file = old_protocol.topology_file
            
            if hasattr(old_protocol, 'trajectory_file'):
                self.protocol.trajectory_file = old_protocol.trajectory_file
            
            logging.info("Switched to MM-PBSA protocol")
            
            return {
                "success": True,
                "message": "Switched to MM-PBSA protocol successfully",
                "previous_protocol": old_protocol.__class__.__name__,
                "current_protocol": "MMPBSAProtocol"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to switch to MM-PBSA protocol: {str(e)}"
            }
        
    def switch_to_protein_ligand_protocol(self) -> Dict[str, Any]:
        """
        Switch to Protein-Ligand protocol
        
        Returns:
            Dictionary with result information
        """
        try:
            # Create new Protein-Ligand protocol
            old_protocol = self.protocol
            self.protocol = ProteinLigandProtocol(self.workspace)
            
            # Copy relevant state from the old protocol if possible
            if hasattr(old_protocol, 'topology_file'):
                self.protocol.topology_file = old_protocol.topology_file
            
            if hasattr(old_protocol, 'trajectory_file'):
                self.protocol.trajectory_file = old_protocol.trajectory_file
            
            logging.info("Switched to Protein-Ligand protocol")
            
            return {
                "success": True,
                "message": "Switched to Protein-Ligand protocol successfully",
                "previous_protocol": old_protocol.__class__.__name__,
                "current_protocol": "ProteinLigandProtocol"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to switch to Protein-Ligand protocol: {str(e)}"
            }
        
    def switch_to_analysis_protocol(self) -> Dict[str, Any]:
        """
        Switch to Analysis protocol
        
        Returns:
            Dictionary with result information
        """
        try:
            # Create new Analysis protocol
            old_protocol = self.protocol
            self.protocol = AnalysisProtocol(self.workspace)
            
            # Copy relevant state from the old protocol if possible
            if hasattr(old_protocol, 'topology_file'):
                self.protocol.topology_file = old_protocol.topology_file
            
            if hasattr(old_protocol, 'trajectory_file'):
                self.protocol.trajectory_file = old_protocol.trajectory_file
            
            logging.info("Switched to Analysis protocol")
            
            return {
                "success": True,
                "message": "Switched to Analysis protocol successfully",
                "previous_protocol": old_protocol.__class__.__name__,
                "current_protocol": "AnalysisProtocol"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to switch to Analysis protocol: {str(e)}"
            }
    
    def get_tool_schema(self) -> List[Dict[str, Any]]:
        """
        Get the schema for the tools available to the LLM
        
        Returns:
            List of tool schema dictionaries
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_shell_command",
                    "description": "Run a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Shell command to run"
                            },
                            "capture_output": {
                                "type": "boolean",
                                "description": "Whether to capture stdout/stderr"
                            }
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_workspace_info",
                    "description": "Get information about the current workspace",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_gromacs_installation",
                    "description": "Check if GROMACS is installed and available",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "set_protein_file",
                    "description": "Set and prepare the protein file for simulation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the protein structure file (PDB or GRO)"
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_for_ligands",
                    "description": "Check for potential ligands in the PDB file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pdb_file": {
                                "type": "string",
                                "description": "Path to the PDB file"
                            }
                        },
                        "required": ["pdb_file"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "set_ligand",
                    "description": "Set the ligand for simulation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ligand_name": {
                                "type": "string",
                                "description": "Residue name of the ligand in the PDB file"
                            }
                        },
                        "required": ["ligand_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_topology",
                    "description": "Generate topology for the protein",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "force_field": {
                                "type": "string",
                                "description": "Name of the force field to use",
                                "enum": ["AMBER99SB-ILDN", "CHARMM36", "GROMOS96 53a6", "OPLS-AA/L"]
                            },
                            "water_model": {
                                "type": "string",
                                "description": "Water model to use",
                                "enum": ["spc", "tip3p", "tip4p"]
                            }
                        },
                        "required": ["force_field"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "define_simulation_box",
                    "description": "Define the simulation box",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "distance": {
                                "type": "number",
                                "description": "Minimum distance between protein and box edge (nm)"
                            },
                            "box_type": {
                                "type": "string",
                                "description": "Type of box",
                                "enum": ["cubic", "dodecahedron", "octahedron"]
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "solvate_system",
                    "description": "Solvate the protein in water",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_mdp_file",
                    "description": "Create an MDP parameter file for GROMACS",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "mdp_type": {
                                "type": "string",
                                "description": "Type of MDP file",
                                "enum": ["ions", "em", "nvt", "npt", "md"]
                            },
                            "params": {
                                "type": "object",
                                "description": "Optional override parameters",
                                "properties": {
                                    "nsteps": {
                                        "type": "integer",
                                        "description": "Number of steps"
                                    },
                                    "dt": {
                                        "type": "number",
                                        "description": "Time step (fs)"
                                    }
                                }
                            }
                        },
                        "required": ["mdp_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "add_ions",
                    "description": "Add ions to the solvated system",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "concentration": {
                                "type": "number",
                                "description": "Salt concentration in M, default is 0.15"
                            },
                            "neutral": {
                                "type": "boolean",
                                "description": "Whether to neutralize the system"
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_energy_minimization",
                    "description": "Run energy minimization",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_nvt_equilibration",
                    "description": "Run NVT equilibration",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_npt_equilibration",
                    "description": "Run NPT equilibration",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_production_md",
                    "description": "Run production MD",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "length_ns": {
                                "type": "number",
                                "description": "Length of the simulation in nanoseconds"
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_rmsd",
                    "description": "Perform RMSD analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_rmsf",
                    "description": "Perform RMSF analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_gyration",
                    "description": "Perform radius of gyration analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_ligand_rmsd",
                    "description": "Perform RMSD analysis focused on the ligand",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_protein_ligand_contacts",
                    "description": "Analyze contacts between protein and ligand",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "set_simulation_stage",
                    "description": "Set the current simulation stage",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "stage": {
                                "type": "string",
                                "description": "Name of the stage to set",
                                "enum": [s.name for s in SimulationStage]
                            }
                        },
                        "required": ["stage"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_mmpbsa_index_file",
                    "description": "Create index file for MM-PBSA analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "protein_selection": {
                                "type": "string",
                                "description": "Selection for protein group"
                            },
                            "ligand_selection": {
                                "type": "string",
                                "description": "Selection for ligand group"
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_mmpbsa_input",
                    "description": "Create input file for MM-PBSA/GBSA calculation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "method": {
                                "type": "string",
                                "description": "Method to use (pb or gb)",
                                "enum": ["pb", "gb"]
                            },
                            "startframe": {
                                "type": "integer",
                                "description": "First frame to analyze"
                            },
                            "endframe": {
                                "type": "integer",
                                "description": "Last frame to analyze"
                            },
                            "interval": {
                                "type": "integer",
                                "description": "Interval between frames"
                            },
                            "ionic_strength": {
                                "type": "number",
                                "description": "Ionic strength for calculation"
                            },
                            "with_entropy": {
                                "type": "boolean",
                                "description": "Whether to include entropy calculation"
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_mmpbsa_calculation",
                    "description": "Run MM-PBSA/GBSA calculation for protein-ligand binding free energy",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ligand_mol_file": {
                                "type": "string",
                                "description": "The Antechamber output mol2 file of ligand parametrization"
                            },
                            "index_file": {
                                "type": "string",
                                "description": "GROMACS index file containing protein and ligand groups"
                            },
                            "topology_file": {
                                "type": "string",
                                "description": "GROMACS topology file (tpr) for the system"
                            },
                            "protein_group": {
                                "type": "string",
                                "description": "Name or index of the protein group in the index file"
                            },
                            "ligand_group": {
                                "type": "string", 
                                "description": "Name or index of the ligand group in the index file"
                            },
                            "trajectory_file": {
                                "type": "string",
                                "description": "GROMACS trajectory file (xtc) for analysis"
                            },
                            "overwrite": {
                                "type": "boolean",
                                "description": "Whether to overwrite existing output files",
                            },
                            "verbose": {
                                "type": "boolean",
                                "description": "Whether to print verbose output",
                            }
                        },
                        "required": ["ligand_mol_file", "index_file", "topology_file", "protein_group", "ligand_group", "trajectory_file"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "parse_mmpbsa_results",
                    "description": "Parse MM-PBSA/GBSA results",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "switch_to_mmpbsa_protocol",
                    "description": "Switch to MM-PBSA protocol for binding free energy calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        ]
        
        return tools
    
    def call_llm(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Call the LLM with messages and tools
        
        Args:
            messages: List of message dictionaries
            tools: List of tool schema dictionaries
            
        Returns:
            LLM response
        """
        tools = tools or self.get_tool_schema()
        
        # 设置请求头，对于本地ollama模型不需要API密钥
        headers = {"Content-Type": "application/json"}
        # 检查是否是本地URL
        is_local_url = "localhost" in self.url or "127.0.0.1" in self.url
        # 只有在非本地URL且有API密钥的情况下添加Authorization头
        if not is_local_url and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        data = {
            "model": self.model,
            "messages": messages,
            "tools": tools
        }
        
        # 实现重试机制
        retries = 0
        while retries <= self.max_retries:
            try:
                # 如果是重试，记录日志
                if retries > 0:
                    wait_time = self.retry_delay * retries
                    logging.info(f"重试连接 LLM API (尝试 {retries}/{self.max_retries})，等待 {wait_time} 秒...")
                    import time
                    time.sleep(wait_time)
                
                response = requests.post(
                    self.url,
                    headers=headers,
                    json=data,
                    timeout=self.request_timeout
                )
                
                if response.status_code != 200:
                    error_msg = f"LLM API 错误: {response.status_code} - {response.text}"
                    logging.error(error_msg)
                    
                    # 如果达到最大重试次数，尝试回退到本地ollama模型
                    if retries == self.max_retries:
                        if self.model != "qwen3:32b" or self.url != "http://localhost:11434/v1/chat/completions":
                            self._handle_api_error(Exception(error_msg))
                            # 重置重试计数并使用新的模型和URL重新调用
                            return self.call_llm(messages, tools)
                        else:
                            raise Exception(error_msg)
                    retries += 1
                    continue
                
                return response.json()
                
            except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError) as e:
                logging.warning(f"连接错误: {str(e)}")
                
                # 如果达到最大重试次数，尝试回退到本地ollama模型
                if retries == self.max_retries:
                    self._handle_api_error(e)
                    # 重置重试计数并使用新的模型和URL重新调用
                    return self.call_llm(messages, tools)
                
                retries += 1
                continue
            except Exception as e:
                logging.error(f"调用 LLM API 时出错: {str(e)}")
                raise
        
        # 如果所有重试都失败
        raise Exception("调用 LLM API 时达到最大重试次数")

    def execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool call
        
        Args:
            tool_call: Tool call dictionary
            
        Returns:
            Result of the tool call
        """
        function_name = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])
        
        if function_name == "set_ligand" and not isinstance(self.protocol, ProteinLigandProtocol):
            # Switch to protein-ligand protocol
            old_protocol = self.protocol
            self.protocol = ProteinLigandProtocol(self.workspace)
            
            # Copy relevant state from the old protocol
            self.protocol.protein_file = old_protocol.protein_file
            self.protocol.stage = old_protocol.stage
            
            logging.info("Switched to protein-ligand protocol")
        elif function_name == "switch_to_mmpbsa_protocol":
            return self.switch_to_mmpbsa_protocol()
        
        # Get the method from the protocol class
        if hasattr(self.protocol, function_name):
            method = getattr(self.protocol, function_name)
            result = method(**arguments)
            return result
        else:
            return {
                "success": False,
                "error": f"Unknown function: {function_name}"
            }
    
    def run(self, starting_prompt: str = None) -> None:
        """
        Run the MD LLM agent
        
        Args:
            starting_prompt: Optional starting prompt for the LLM
        """
        # Initialize conversation with system message
        if self.mode == "copilot":
            system_message = {
                "role": "system",
                "content": SYSTEM_MESSAGE_ADVISOR
            }
        else:
            system_message = {
                "role": "system",
                "content": SYSTEM_MESSAGE_AGENT
            }
        
        self.conversation_history = [system_message]
        
        # Add starting prompt if provided
        if starting_prompt:
            self.conversation_history.append({
                "role": "user",
                "content": starting_prompt
            })
        
        # Get initial response from LLM
        response = self.call_llm(self.conversation_history)
        
        # Main conversation loop
        while True:
            assistant_message = response["choices"][0]["message"]
            self.conversation_history.append(assistant_message)
            
            # Process tool calls if any
            if "tool_calls" in assistant_message:
                for tool_call in assistant_message["tool_calls"]:
                    # Execute the tool call
                    print_message(f"Executing: {tool_call['function']['name']}", MessageType.TOOL)
                    result = self.execute_tool_call(tool_call)
                    
                    # Add the tool call result to the conversation
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "content": json.dumps(result)
                    })
                
                # Get next response from LLM
                response = self.call_llm(self.conversation_history)
                continue
            
            # Display the assistant's message
            content = assistant_message["content"]
            
            # Check if it's a final answer
            if "This is the final answer at this stage." in content:
                # Split at the final answer marker
                parts = content.split("This is the final answer at this stage.")
                
                # Print the main content normally
                print_message(parts[0].strip(), MessageType.INFO)
                
                # Print the final answer part with special formatting
                final_part = "This is the final answer at this stage." + parts[1]
                print_message(final_part.strip(), MessageType.FINAL, style="box")
            else:
                # Regular message
                print_message(content, MessageType.INFO)
            
            # Check if we've reached a stopping point
            if "This is the final answer at this stage." in content:
                # Ask if the user wants to continue
                user_input = prompt_user("Do you want to continue with the next stage?", default="yes")
                if user_input.lower() not in ["yes", "y", "continue", ""]:
                    print_message("Exiting the MD agent. Thank you for using GROMACS Copilot!", MessageType.SUCCESS, style="box")
                    break
                
                # Ask for the next user prompt
                user_input = prompt_user("What would you like to do next?")
            else:
                # Normal user input
                user_input = prompt_user("Your response")
            
            # Check for exit command
            if user_input.lower() in ["exit", "quit", "bye"]:
                print_message("Exiting the MD agent. Thank you for using GROMACS Copilot!", MessageType.SUCCESS, style="box")
                break
            
            # Add user input to conversation
            self.conversation_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Get next response from LLM
            response = self.call_llm(self.conversation_history)