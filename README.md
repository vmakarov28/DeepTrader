# Deep Trader <sup>10<sup>

This is v10.00.00 of the project formly known as Alpaca Neural Bot (v0-v9). DeepTrader is an AI-powered stock trading system that uses deep learning to predict short-term price moves and automatically place trades through the Alpaca API. Built with PyTorch and highly optimized for NVIDIA GPUs, it combines technical analysis, market regime detection, and strict risk controls to make trading decisions.

This project handles everything from data fetching and model training to testing, execution and notifications.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-812CE5?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Alpaca API](https://img.shields.io/badge/Alpaca%20API-Excecution-00BFFF?style=flat&logo=alpaca&logoColor=white)](https://alpaca.markets/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![TA-Lib](https://img.shields.io/badge/TA--Lib-Indicators-FF6F00?style=flat-square&logo=python&logoColor=white)](https://ta-lib.org/)

---

### Key Features

- **Neural Network Prediction**: LSTM model with Multihead Attention that analyzes 30-timestep sequences of **31 features** (including RSI, MACD, ATR, ADX, volume profile, multi-timeframe indicators, and earnings/sentiment proxies) to forecast price direction over the next 21 bars (~5 hours at 15-minute intervals).
- **Market Regime Detection**: Hidden Markov Model (HMM) that identifies 6 different market states (Calm Bull, Volatile Bull, etc.) to improve signal quality.
- **Ensemble Power**: Combines LSTM predictions with XGBoost for more reliable buy/sell decisions.
- **Pairs Trading**: Built-in market-neutral strategy using cointegrated pairs (AAPL-MSFT, NVDA-AMD, etc.) with spread and z-score logic.
- **Strong Risk Management**: ATR-based stop-loss and take-profit, trailing stops, volatility filters, RSI/ADX thresholds, maximum drawdown protection, and minimum holding periods.
- **Advanced Backtesting**: Automatically runs multiple training attempts, selects the absolute best models per symbol, and reports Sharpe ratio, max drawdown, win rate, accuracy, Monte Carlo simulations, and direct comparison to Buy-and-Hold stratiges.
- **Live & Paper Trading**: Executes real market orders during market hours, includes real-time regime detection, and sends email alerts for every trade plus daily summaries.
- **Performance Graphing**: Shows three lines — **Day Trading equity curve (blue)**, **Buy-and-Hold (green)**, and a dashed red **Initial Cash breakeven line** — so you instantly see when you're in profit and how alternative stratigies preform.

---

### How It Works

The Alpaca Neural Bot follows a three-stage process: **training & backtesting** first, then **live trading**.

### 1. Training & Backtesting Phase
1. **Data Collection** — Downloads years of 15-minute historical bars from the Alpaca API (with smart caching and retries).
2. **Feature Engineering** — Calculates 31 technical indicators (RSI, MACD, ATR, ADX, Bollinger Bands, volume profile, multi-timeframe data, etc.) plus sentiment.
3. **Model Training** — Trains an LSTM + Multihead Attention neural network on GPU for each symbol. Also builds a Hidden Markov Model (HMM) for market regime detection and an XGBoost ensemble on CPU for voting style decision making.
4. **Automated Optimization** — Runs repeated full training attempts, backtests each one, and automatically keeps the best-performing models per symbol.
5. **Realistic Simulation** — Performs detailed backtesting with ATR-based stops, trailing stops, volatility filters, pairs trading, transaction costs, and Monte Carlo simulations. Generates a clear graph comparing Day Trading to Buy-and-Hold.

### 2. Live Trading Phase
Once the best models are ready:
1. Loads the trained models and scalers from disk.
2. Every 15 minutes (while the market is open):
   - Fetches the latest price data
   - Updates all 31 indicators and detects the current market regime
   - Runs the LSTM + XGBoost ensemble to generate a confidence-weighted prediction
   - Applies strict risk filters (confidence threshold, RSI, ADX, volatility)
   - Decides **Buy**, **Sell**, or **Hold**
   - Executes market orders through Alpaca
   - Sends real-time email alerts for every trade and daily summaries

The system continuously monitors portfolio drawdown and enforces conservative position sizing and risk rules at all times.

### Current Status (March 2026)

- Supports 8 major stocks + 4 pairs
- Fully working live/paper trading with email alerts
- Continuous daily equity tracking (no more flat blue line!)
- Optimized for Alpaca’s free tier (caching, retries, rate-limit friendly)
- Runs on consumer hardware with an RTX 50 and 40 series

---

**Note**: This is an educational and research project. Always backtest thoroughly, start with paper trading, and never risk money you can’t afford to lose.

---

### Simulated BackTest Results from 1/1/2025 to 3/19/2026
<img width="1200" height="600" alt="After Changes" src="https://github.com/user-attachments/assets/c16be814-5ef5-4a6a-92a6-44d8bacb93ae" />
Simulated BackTest Results from 1/1/2025 to 3/13/2026

### Real Life Long Duration Paper Trading Results (Last Updated 2/7/26)
<img width="979" height="251" alt="image" src="https://github.com/user-attachments/assets/13649f49-4902-47fe-8887-311688e55c17" />

- Starting Value: $100,000
  
- Active Days Running: 122
  
- Current Portfolio Value: $128,613.36
  
- Lowest Recorded Portfolio Value: $97,185.30 

Disclaimer: ***It is HIGHLY recommened to use this for educational purposes only. Use paper trading to avoid real financial risk.***

---

# Installation Steps
Follow these steps in order to set up the environment. 

**Note:** All commands after step 2 are for WSL/Ubuntu terminal.

## Prerequisites

- Drivers: Install the latest nvidia drivers from nvdia app for your GPU.
- Hardware: NVIDIA GPU with at least 16GB VRAM for efficient training. (Confirmed to work properly on RTX 5080 but other 50 series and 40 series cards are likely to work).
- Operating System: Windows Subsystem for Linux (WSL2) on Windows 11 or Native Ubuntu 22.04+
- Alpaca Account: Free account with paper trading enabled. Get API keys from Alpaca Dashboard.
- Gmail Account: For email notifications (enable "Less secure app access" or use app password).
- Internet: Stable connection for API calls (VPN is not recommended due to possible rate limits).



### Step 1: Set Up WSL (Windows Users Only)

If you're on Windows, enable WSL2 for Linux-based setup. Open PowerShell as Administrator and install Windows Subsystem for Linux:
```
wsl --install
wsl --install -d Ubuntu
wsl --set-default-version 2
sudo apt update && sudo apt upgrade -y
```

Restart your PC after successful installation.

### Troubleshooting:
- If WSL installation fails, enable "Virtual Machine Platform" in Windows Features (Control Panel > Programs > Turn Windows features on/off).
- Check WSL version: wsl --list --verbose. Ensure Ubuntu is on version 2.

## Step 2: Install pyenv for Python Management
*********\*\***From now on the rest of the steps will be completed in wsl*\*\***********

We use Pyenv because it allows isolated Python environments.

### Install dependencies for Pyenv:
Use this command:

    sudo apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git


Install pyenv:

    curl https://pyenv.run | bash

Add to shell profile (add to ~/.bashrc or ~/.zshrc). Open the file with the nano text editor with this command:

    nano ~/.bashrc
    
Then add the new paths to the end of the file.
> export PATH="$HOME/.pyenv/bin:$PATH"
> 
> eval "$(pyenv init --path)"
> 
> eval "$(pyenv virtualenv-init -)"
> 
Press `Cntrl + O` then `Enter` to save. Then use `Cntrl + X` to exit the file.
Now back in the command line refresh the file with:

    source ~/.bashrc
Verify Pyenv was setup correcty with:

    pyenv --version
Expected output: `pyenv 2.6.7` (Latest as of 8/24/25)
### Troubleshooting:
- If "Unable to locate package llvm" use:
  
        sudo add-apt-repository universe
        sudo apt update
  then re-attempt the command:

      sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git
- If pyenv not found, restart terminal or run exec $SHELL.
- Error with dependencies: Re-run sudo apt install command.

## Step 3: Set Up Python Environment

Use Python 3.10.12 (compatible with dependencies).
Install Python 3.10:

    pyenv install 3.10.12
Create virtual environment:

    pyenv virtualenv 3.10.12 pytorch_env
Activate the virtual envoriment (You will also need this command everytime you want to run the program after exiting:

    pyenv activate pytorch_env
Upgrade pip:

    pip install --upgrade pip

Troubleshooting
- If you get ``pyenv activate' requires Pyenv and Pyenv-Virtualenv to be loaded into your shell.
Check your shell configuration and Pyenv and Pyenv-Virtualenv installation instructions.`` When using the command `pyenv activate pytorch_env` Run each line individually:

        echo 'eval "$(pyenv init -)"' >> ~/.bashrc
        eval "$(pyenv init -)"
        source ~/.bashrc
  then reattempt `pyenv activate pytorch_env`
- If build fails, ensure all dependencies from Step 1 are installed.
- Syntax Errors: Make sure all the paths are written properly with no mistakes.

## Step 4: Install CUDA Toolkit in WSL

Open your WSL terminal (e.g., Ubuntu): Run wsl in Command Prompt or search for "Ubuntu" in the Start menu.
Update the package list:

    sudo apt update && sudo apt upgrade -y
    
Install the CUDA toolkit for WSL-Ubuntu:
Add the NVIDIA CUDA repository: Follow the instructions at https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0.
> Select "WSL-Ubuntu" > "2.0" > "x86_64"

Then copy and run the provided commands:

        wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        sudo apt-get update
        sudo apt-get install cuda-toolkit-12-6  # Replace 12-6 with the latest version, e.g., 12-5 if needed
Add new paths to bash by running these commmands one by one:

        echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
        echo 'export PATH=/usr/lib/wsl/lib:$PATH' >> ~/.bashrc
        source ~/.bashrc

Verify CUDA installation: Run `nvcc --version` in WSL. It should display the CUDA version.

## Step 5: Install cuDNN in WSL

Download cuDNN from the NVIDIA Developer website: https://developer.nvidia.com/rdp/cudnn-download.
- You made need to sign up for an NVIDIA Developer account if you don't have one (free).
- Select cuDNN for CUDA 12.x (matching your toolkit version) and "Linux" (x86_64).
- Select Linux > x86_64 > Ubuntu > 24.04 (Yours may varry) > deb (local) > FULL
- Download the deb file (e.g., cudnn-local-repo-ubuntu2404-9.12.0_1.0-1_amd64.deb).
    In WSL, extract and install cuDNN:
        Copy the downloaded file to WSL (e.g., via /mnt/c/Users/YourUsername/Downloads/).
  
Run each command sequentially (**in pyenv**:
```
cd /mnt/c/Users/YOUR USER/Downloads
sudo dpkg -i cudnn-local-repo-ubuntu2404-9.12.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2404-9.12.0/cudnn-local-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libcudnn9-static-cuda-12
echo 'export LD_LIBRARY_PATH=/usr/lib/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Troubleshooting
- Run:

      wget https://developer.download.nvidia.com/compute/cudnn/9.12.0/local_installers/cudnn-local-repo-ubuntu2404-9.12.0_1.0-1_amd64.deb
      sudo dpkg -i cudnn-local-repo-ubuntu2404-9.12.0_1.0-1_amd64.deb
      sudo cp /var/cudnn-local-repo-ubuntu2404-9.12.0/cudnn-*-keyring.gpg /usr/share/keyrings/
      sudo apt-get update
      sudo apt-get -y install cudnn
- If CUDA is not detected: Check nvidia-smi in WSL for GPU info.
- Errors with versions: Ensure CUDA toolkit, cuDNN, and PyTorch match (e.g., all for CUDA 12.x).
- For detailed guides: Refer to NVIDIA's CUDA on WSL user guide (linked above) or PyTorch installation docs at https://pytorch.org/get-started/locally/.



    

## Step 6: Install PyTorch with confirm CUDA Support

For RTX 50 & 40 series GPU acceleration we will PyTorch for training the AI model (use cu128 for CUDA 12.8; adjust if your version differs):

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 

Verify PyTorch, CUDA, and GPU detection:

    python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
Expected: Shows PyTorch version (e.g., 2.4.1+cu128), True, and "NVIDIA GeForce RTX _ _ _ _".

### Troubleshooting


- False for CUDA: Ensure NVIDIA drivers are installed.
- Wrong device: Verify nvidia-smi shows RTX 5080.
- Installation error: Check PyTorch index URL for CUDA version compatibility.

## Step 7: Install TA-Lib

TA-Lib requires building from source for technical indicators.

Install dependencies:

    sudo apt install -y build-essential wget

Download and build TA-Lib:
Remove any previous build artifacts (optional, to clean)
    
    rm -rf ~/ta-lib ta-lib-0.4.0-src.tar.gz
Re-download and build with /usr/local prefix

    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib
    ./configure --prefix=/usr/local
    make
    sudo make install
    cd ~
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
    
Set environment variables for pip to find TA-Lib

    export TA_INCLUDE_PATH=/usr/local/include
    export TA_LIBRARY_PATH=/usr/local/lib
    
Install the Python wrapper
    pip install TA-Lib==0.4.32

Install Python wrapper:

    pip install TA-Lib==0.4.32

Troubleshooting

- Build error: Ensure build-essential is installed. If configure fails, check for missing libraries (e.g., sudo apt install libncurses5-dev). Another option it to try and download a prebuilt wheel from the Ta-Lib repo.
- Import error: Verify with python -c "import talib; print(talib.__version__)".

## Step 8: Install Other Dependencies

Install the remaining packages with exact versions for compatibility.
Install dependencies:

    pip install alpaca-py==0.28.0 transformers==4.45.2 pandas==2.2.3 numpy==1.26.4 scikit-learn==1.5.2 tenacity==9.0.0 tqdm==4.66.5 colorama==0.4.6 protobuf==5.28.3

Verify imports:

    python -c "import torch, numpy, pandas, alpaca, transformers, sklearn, talib, tenacity, tqdm, colorama; print('All dependencies imported successfully')"

Troubleshooting:

    Conflicts: Use --no-deps for problematic packages (e.g., pip install alpaca-py==0.28.0 --no-deps).
    Protobuf error: Ensure protobuf==5.28.3 (for Transformers compatibility).
    Import error: Reinstall the package (e.g., pip install --force-reinstall transformers==4.45.2).


## Step 9: Configure API Keys and Email

- Replace placeholders in CONFIG (lines 147-148, 153-156) with your values:
- ALPACA_API_KEY and ALPACA_SECRET_KEY from Alpaca.
- EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER for Gmail.
- Save the script.

## Step 10: Running the Script
*Note:* If you do not want to train your own models / scalars, You can download already trained models & scalars, put them in a folder, amd put the address into the 'Config':
Activate virtual environment:

    pyenv activate pytorch_env

Run backtest with force-train

    python /mnt/c/Users/aipla/Downloads/alpaca_neural_bot_v6.7.py --backtest --force-train

Run live trading:

    python /mnt/c/Users/aipla/Downloads/alpaca_neural_bot_v6.7.py

### Cool Facts about the hardware
- The raw compute happening on the RTX 5080 during training is equivalent to running ~85,000 PlayStation 5 consoles all at full tilt doing nothing but matrix multiplications.
- If each floating-point operation were a grain of sand, in one second the GPU would process enough sand to fill an Olympic swimming pool … every 45 seconds.
- In one second, this program flips more transistors states (1s/0s) than there are stars in the observable universe (~2 × 10¹² galaxies × ~10¹¹ stars each) … 4,000 times over.
- The CPU is moving enough data through RAM to stream ~23 000 simultaneous 4K Netflix videos.
- In total, the computer switches 28 quintillion (28,000,000,000,000,000,000) transistor state changes per second

## Troubleshooting

- Rate Limit Error ("too many requests"): Add time.sleep(1) after bars = client.get_stock_bars(request).df in fetch_data (line 287). Increase to 2 seconds if persists.
- Missing Headers (longintrepr.h): Ensure apt-get install -y python3.12-dev ran successfully. Verify with find /usr/include -name longintrepr.h.
- Wheel Not Supported: Use Python 3.10-compatible wheels. For aiohttp, try `pip install aiohttp==3.9.5 if 3.8.1 fails`.
- CUDA Not Available: Check nvidia-smi. Reinstall drivers/CUDA. Verify with python -c "import torch; print(torch.cuda.is_available())".
- Dependency Conflicts: Use --no-deps for alpaca-py (e.g., `pip install alpaca-py==0.28.0 --no-deps`). Uninstall conflicting packages (e.g., `pip uninstall aiohttp`).
- TA-Lib Import Error: Rebuild TA-Lib and reinstall wrapper. Verify with `python -c "import talib; print(talib.__version__)"`.
- API Key Error: Ensure keys are valid. Test with `python -c "from alpaca.trading.client import TradingClient; client = TradingClient('YOUR_KEY', 'YOUR_SECRET', paper=True); print(client.get_account())"`.
- Model Training Hangs: Reduce batch size in CONFIG (line 75) to 16 if GPU memory is low. Monitor with nvidia-smi.
- Backtest IndexError: Clear cache with `rm -rf ./cache` and re-run to fetch data up to today.
- Email Failure: Ensure Gmail app password is used (not regular password). Test SMTP with a simple script.
Ensure all dependency files are installed:
    `pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`
    `pip install --upgrade alpaca-py transformers pandas numpy scikit-learn ta-lib tenacity tqdm colorama protobuf==5.28.3`

If issues persist, check trades.log for details or open an issue on the GitHub repo.


License
GNU Lesser General Public License v2.1. See LICENSE for details.

Author: Vladimir Makarov

Most recent change: 5/22/2026

GitHub: vmakarov28/Alpaca-Stock-Trading-Bot
