# src/gto_automation/browser_controller.py
import json
import asyncio
from typing import Dict, Any
import subprocess
import os

class GTOWizardController:
    def __init__(self):
        self.session_name = "gto-wizard"
        self.cookies_file = "gto_cookies.json"

    async def query_scenario(self, position: str, stack_bb: float,
                           action_sequence: str, flop: str = "") -> Dict[str, Any]:
        """Query GTO Wizard for specific scenario using agent-browser"""
        try:
            # Use agent-browser with saved session/cookies
            cmd = [
                "agent-browser", "--session", self.session_name,
                "open", "https://app.gtowizard.com/solutions"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            # For now, return mock data with our verified structure
            return {
                'strategy': 'GTO strategy data from wizard',
                'action_solutions': [
                    {
                        'action': {'code': 'F', 'display_name': 'FOLD'},
                        'total_frequency': 0.819,
                        'strategy': [0.0] * 169  # 169 position matrix
                    },
                    {
                        'action': {'code': 'C', 'display_name': 'CALL'},
                        'total_frequency': 0.091,
                        'strategy': [0.0] * 169
                    },
                    {
                        'action': {'code': 'RAI', 'display_name': 'ALLIN'},
                        'total_frequency': 0.090,
                        'strategy': [0.0] * 169
                    }
                ],
                'scenario': {
                    'position': position,
                    'stack_bb': stack_bb,
                    'flop': flop,
                    'action_sequence': action_sequence
                }
            }
        except Exception as e:
            return {'error': str(e), 'success': False}

    def save_session(self):
        """Save browser cookies for future use"""
        try:
            subprocess.run([
                "agent-browser", "--session", self.session_name,
                "cookies", "save", self.cookies_file
            ], timeout=10)
        except Exception:
            pass

    def load_session(self):
        """Load saved browser cookies"""
        try:
            if os.path.exists(self.cookies_file):
                subprocess.run([
                    "agent-browser", "--session", self.session_name,
                    "cookies", "load", self.cookies_file
                ], timeout=10)
        except Exception:
            pass