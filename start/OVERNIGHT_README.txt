# Copy to prism-engine
cp /path/to/run_overnight_analysis.py ~/prism-engine/start/

# Run in background (won't stop if you close terminal)
cd ~/prism-engine
source venv/bin/activate
nohup python start/run_overnight_analysis.py > overnight.log 2>&1 &

# Check progress anytime
tail -f overnight.log