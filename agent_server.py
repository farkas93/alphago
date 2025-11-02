# agent_server.py
"""
AI Agent WebSocket Client for Go Game Platform
Connects to the game server and plays as an AI opponent
"""
import asyncio
import websockets
import json
import numpy as np
import torch
from typing import Optional, Dict, Any
import argparse
import sys

from agents.policy_agent import PolicyAgent
from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from game.go_board import GoBoard
from config import BOARD_SIZE


class AgentClient:
    """WebSocket client that controls an AI agent"""
    
    def __init__(self, agent, agent_name: str, server_url: str = "ws://localhost:3000/api/socket"):
        self.agent = agent
        self.agent_name = agent_name
        self.server_url = server_url
        self.ws = None
        self.player_id = None
        self.current_session = None
        self.board = None
        
    async def connect(self):
        """Establish WebSocket connection"""
        try:
            self.ws = await websockets.connect(self.server_url)
            print(f"[AgentClient] Connected to {self.server_url}")
            return True
        except Exception as e:
            print(f"[AgentClient] Connection failed: {e}")
            return False
    
    async def register(self):
        """Register the agent as a player"""
        message = {
            "type": "register",
            "payload": {
                "name": self.agent_name,
                "type": "ai"  # Mark as AI player
            }
        }
        await self.ws.send(json.dumps(message))
        print(f"[AgentClient] Registered as {self.agent_name}")
    
    async def find_match(self, game_type: str = "go"):
        """Request matchmaking for Go game"""
        message = {
            "type": "find_match",
            "payload": {
                "gameType": game_type
            }
        }
        await self.ws.send(json.dumps(message))
        print(f"[AgentClient] Looking for {game_type} match...")
    
    def convert_server_state_to_board(self, state: Dict[str, Any]) -> GoBoard:
        """
        Convert server's game state to GoBoard instance
        Server state format matches GoGame.ts structure
        """
        board = GoBoard(BOARD_SIZE)
        
        # Convert board representation
        # Server: null/string array, Agent: 0/1/-1 numpy array
        server_board = state.get('board', [])
        for row in range(len(server_board)):
            for col in range(len(server_board[row])):
                cell = server_board[row][col]
                if cell == 'black':
                    board.board[row, col] = 1
                elif cell == 'white':
                    board.board[row, col] = -1
                else:
                    board.board[row, col] = 0
        
        # Set current player
        current_turn = state.get('currentTurn', 'black')
        board.current_player = 1 if current_turn == 'black' else -1
        
        # Set captures
        captures = state.get('captures', {'black': 0, 'white': 0})
        board.captured = {1: captures['black'], -1: captures['white']}
        
        # Ko point
        ko_point = state.get('koPoint')
        if ko_point:
            board.ko_point = (ko_point['row'], ko_point['col'])
        
        return board
    
    def get_my_color(self, session: Dict[str, Any]) -> str:
        """Determine which color this agent is playing"""
        players = session.get('players', [])
        for idx, player in enumerate(players):
            if player.get('id') == self.player_id:
                # First player is black, second is white
                return 'black' if idx == 0 else 'white'
        return 'black'  # Default
    
    def is_my_turn(self, state: Dict[str, Any], my_color: str) -> bool:
        """Check if it's this agent's turn"""
        current_turn = state.get('currentTurn', 'black')
        return current_turn == my_color
    
    async def make_move(self, session_id: str, row: int, col: int):
        """Send move to server"""
        message = {
            "type": "move",
            "payload": {
                "sessionId": session_id,
                "move": {
                    "row": row,
                    "col": col
                }
            }
        }
        await self.ws.send(json.dumps(message))
        print(f"[AgentClient] Played move at ({row}, {col})")
    
    async def handle_game_start(self, session: Dict[str, Any]):
        """Handle game start event"""
        self.current_session = session
        print(f"[AgentClient] Game started! Session ID: {session['id']}")
        print(f"[AgentClient] Players: {[p['name'] for p in session['players']]}")
        
        my_color = self.get_my_color(session)
        print(f"[AgentClient] Playing as {my_color}")
        
        # If we're black (first player), make first move
        if my_color == 'black':
            await self.think_and_move(session)
    
    async def handle_game_update(self, session: Dict[str, Any]):
        """Handle game state update"""
        self.current_session = session
        state = session.get('state', {})
        
        # Check if game ended
        if state.get('ended'):
            winner = state.get('winner')
            print(f"[AgentClient] Game over! Winner: {winner}")
            return
        
        my_color = self.get_my_color(session)
        
        # Check if it's our turn
        if self.is_my_turn(state, my_color):
            print(f"[AgentClient] My turn ({my_color})")
            await self.think_and_move(session)
        else:
            print(f"[AgentClient] Opponent's turn")
    
    async def think_and_move(self, session: Dict[str, Any]):
        """Use agent to select and play a move"""
        state = session.get('state', {})
        
        # Convert to GoBoard
        board = self.convert_server_state_to_board(state)
        
        # Let agent think
        print(f"[AgentClient] Agent thinking...")
        move = self.agent.select_move(board)
        
        if move is None:
            print(f"[AgentClient] No legal moves available (passing)")
            # TODO: Implement pass functionality
            return
        
        row, col = move
        
        # Small delay to make it feel more natural
        await asyncio.sleep(0.5)
        
        # Send move to server
        await self.make_move(session['id'], row, col)
    
    async def handle_message(self, message: Dict[str, Any]):
        """Route incoming messages to appropriate handlers"""
        msg_type = message.get('type')
        payload = message.get('payload', {})
        
        if msg_type == 'registered':
            self.player_id = payload['player']['id']
            print(f"[AgentClient] Registered with ID: {self.player_id}")
            # After registration, find a match
            await self.find_match('go')
            
        elif msg_type == 'waiting_for_opponent':
            print(f"[AgentClient] Waiting for opponent...")
            
        elif msg_type == 'game_start':
            await self.handle_game_start(payload['session'])
            
        elif msg_type == 'game_update':
            await self.handle_game_update(payload['session'])
            
        elif msg_type == 'game_end':
            result = payload.get('result', {})
            print(f"[AgentClient] Game ended: {result}")
            # Could auto-search for new game here
            
        elif msg_type == 'error':
            print(f"[AgentClient] Error: {payload.get('message')}")
    
    async def run(self):
        """Main event loop"""
        if not await self.connect():
            return
        
        try:
            # Register player
            await self.register()
            
            # Listen for messages
            async for message in self.ws:
                try:
                    data = json.loads(message)
                    print(f"[AgentClient] Received: {data['type']}")
                    await self.handle_message(data)
                except json.JSONDecodeError as e:
                    print(f"[AgentClient] Failed to parse message: {e}")
                except Exception as e:
                    print(f"[AgentClient] Error handling message: {e}")
                    import traceback
                    traceback.print_exc()
                    
        except websockets.exceptions.ConnectionClosed:
            print("[AgentClient] Connection closed")
        except KeyboardInterrupt:
            print("\n[AgentClient] Shutting down...")
        finally:
            if self.ws:
                await self.ws.close()


def create_agent(agent_type: str):
    """Factory function to create different agent types"""
    if agent_type == 'random':
        return RandomAgent()
    elif agent_type == 'heuristic':
        return HeuristicAgent()
    elif agent_type == 'policy':
        return PolicyAgent()
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


async def main():
    parser = argparse.ArgumentParser(description='AI Agent for Go Game Server')
    parser.add_argument('--agent', type=str, default='heuristic',
                       choices=['random', 'heuristic', 'policy'],
                       help='Type of agent to use')
    parser.add_argument('--name', type=str, default=None,
                       help='Agent name (default: auto-generated)')
    parser.add_argument('--server', type=str, default='ws://localhost:3000/api/socket',
                       help='WebSocket server URL')
    
    args = parser.parse_args()
    
    # Create agent
    agent = create_agent(args.agent)
    agent_name = args.name or f"{args.agent.capitalize()} Bot"
    
    print(f"[AgentServer] Starting {agent_name}")
    print(f"[AgentServer] Server: {args.server}")
    
    # Create and run client
    client = AgentClient(agent, agent_name, args.server)
    await client.run()


if __name__ == "__main__":
    asyncio.run(main())