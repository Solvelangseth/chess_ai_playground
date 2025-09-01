# Chess AI Playground

A sandbox environment for experimenting with different chess AI approaches, from traditional minimax algorithms to neural networks trained on master games. The project serves as a testbed for comparing various AI techniques and exploring reinforcement learning applications in chess.

## Project Overview

This playground implements multiple chess AI approaches:

1. **Traditional Engines**: Two minimax-based AIs with increasing sophistication
2. **Neural Network Foundation**: CNN architecture for position evaluation
3. **Data Pipeline**: Automated conversion of PGN game files to training tensors
4. **Future RL Training**: Framework for training neural networks against traditional AIs

## Architecture

### Traditional Chess Engines

**ChessAI (Basic Engine)**
- Minimax with alpha-beta pruning
- Simple piece-square tables
- Material + mobility evaluation
- Transposition table caching

**ChessAI2 (Advanced Engine)**
- Sophisticated position evaluation
- Opening book integration
- Pawn structure analysis
- King safety (midgame vs endgame)
- Quiescence search for tactics
- Move ordering optimizations

### Neural Network Components

**CNN Architecture** (`neural_net_evaluator.py`)
- Input: 12x8x8 tensors (6 piece types × 2 colors)
- Convolutional layers for pattern recognition
- Output: Position evaluation (-1 to 1)

**Data Generation Pipeline** (`generating_data.py`)
- Processes PGN files from master games
- Uses Stockfish for ground truth evaluations
- Converts positions to tensor format
- Parallel processing for large datasets

### Game Interface

**Visual Chess Game** (`game.py`, `main.py`)
- PyGame-based GUI for human vs AI play
- Real-time move visualization
- Piece highlighting and move indication

**UCI Protocol** (`uci.py`)
- Standard chess engine interface
- Compatible with chess GUIs like Arena or ChessBase

## Installation

```bash
pip install python-chess pygame torch stockfish
```

**Stockfish Setup:**
- Install Stockfish engine
- Update `STOCKFISH_PATH` in `generating_data.py`

**Asset Requirements:**
Place chess piece images in `assets/` directory:
```
assets/white-pawn.png
assets/black-queen.png
... (all pieces)
```

## Usage

### Play Against AI
```bash
python main.py
```
Click to select and move pieces. The AI (ChessAI) will respond automatically.

### AI vs AI Tournament
```bash
python play_game.py
```
Runs a tournament between ChessAI and ChessAI2 with performance metrics.

### Generate Training Data
```bash
python generating_data.py
```
Processes PGN files in `data/` folder, outputs tensors to `tensor_data/`.

### UCI Engine Mode
```bash
python uci.py
```
Runs the AI as a UCI-compatible engine for external chess GUIs.

### Debug Training Data
```bash
python debug.py
```
Inspects the structure of generated tensor files.

## File Structure

```
├── ai.py                    # Basic minimax AI
├── ai2.py                   # Advanced chess engine
├── game.py                  # PyGame chess interface
├── main.py                  # Game launcher
├── play_game.py             # AI tournament system
├── generating_data.py       # PGN to tensor conversion
├── neural_net_evaluator.py  # CNN model definition
├── uci.py                   # UCI protocol interface
├── debug.py                 # Training data inspection
├── settings.py              # Game configuration
├── pieces.py                # Piece image loading
├── board.py                 # Board representation
├── data/                    # PGN files for training
├── tensor_data/             # Generated training tensors
└── assets/                  # Chess piece images
```

## Planned Developments

### Neural Network Training
- Train CNN on master game positions
- Compare neural network vs traditional evaluation
- Hybrid approaches combining both methods

### Reinforcement Learning
- Self-play training framework
- Neural network vs traditional AI matches
- Policy gradient methods for move selection
- AlphaZero-style training pipeline

### Advanced Features
- Monte Carlo Tree Search integration
- Opening book expansion
- Endgame tablebase integration
- Performance profiling and optimization

## AI Comparison

Current engine capabilities:

| Feature | ChessAI | ChessAI2 | Neural Net (Planned) |
|---------|---------|----------|----------------------|
| Search Depth | 3-5 | 4+ | Variable |
| Evaluation | Basic | Advanced | Learned |
| Opening Book | No | Yes | Trained |
| Endgame Knowledge | Limited | Improved | Learned |
| Tactical Awareness | Alpha-Beta | Quiescence | Pattern Recognition |

## Performance Notes

- ChessAI2 typically outperforms ChessAI in tournaments
- Search depth vs time tradeoffs configurable
- Neural network training requires significant compute resources
- PGN processing is parallelized for efficiency

## Contributing

This is a research and learning project. The codebase is designed for experimentation with different chess AI approaches rather than production use.
