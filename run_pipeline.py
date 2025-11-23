#!/usr/bin/env python3
"""
Master Pipeline Runner for FACEIT CS2 Player Classifier

This script runs the complete ML pipeline from data collection to model training.
You can run individual steps or the entire pipeline.
"""

import os
import sys
import subprocess
import time
from datetime import datetime

# ANSI color codes for pretty output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.END}\n")

def print_step(step_num, title):
    print(f"\n{Colors.CYAN}{Colors.BOLD}📌 STEP {step_num}: {title}{Colors.END}")
    print(f"{Colors.CYAN}{'─'*70}{Colors.END}\n")

def print_success(message):
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_error(message):
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_warning(message):
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_info(message):
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.END}")

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print_info(f"Running: {script_name}")
    print_info(f"Description: {description}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        print_success(f"Completed in {elapsed:.1f} seconds")
        return True
    
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to run {script_name}")
        print_error(f"Error: {e}")
        return False
    except FileNotFoundError:
        print_error(f"Script not found: {script_name}")
        return False

def check_file_exists(filepath):
    """Check if a required file exists"""
    if os.path.exists(filepath):
        print_success(f"Found: {filepath}")
        return True
    else:
        print_warning(f"Missing: {filepath}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ['data', 'models', 'results', 'templates']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print_success(f"Directory ready: {directory}/")

def check_dependencies():
    """Check if required packages are installed"""
    print_step(0, "CHECKING DEPENDENCIES")
    
    required_packages = [
        'requests', 'numpy', 'pandas', 'sklearn', 
        'xgboost', 'tensorflow', 'flask', 'tqdm'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print_success(f"Package installed: {package}")
        except ImportError:
            missing.append(package)
            print_error(f"Package missing: {package}")
    
    if missing:
        print_error(f"\nMissing packages: {', '.join(missing)}")
        print_info("Run: pip install -r requirements.txt")
        return False
    
    print_success("\nAll dependencies installed!")
    return True

def run_full_pipeline():
    """Run the complete pipeline"""
    print_header("🚀 FACEIT CS2 PLAYER CLASSIFIER - FULL PIPELINE")
    print_info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check dependencies
    if not check_dependencies():
        print_error("\n⚠️  Please install missing dependencies first!")
        return False
    
    # Create directories
    print_step(1, "CREATING DIRECTORIES")
    create_directories()
    
    # Define pipeline steps
    pipeline_steps = [
        {
            "num": 2,
            "script": "pro_players.py",
            "title": "COLLECT PRO PLAYERS FROM HLTV",
            "description": "Scraping professional player data",
            "required_files": [],
            "output_files": ["pro_faceit_data.csv"],
            "estimated_time": "5-10 minutes"
        },
        {
            "num": 3,
            "script": "data_collection.py",
            "title": "COLLECT PLAYER IDs BY CLASS",
            "description": "Gathering PRO, HIGH-LEVEL, and NORMAL player IDs",
            "required_files": ["pro_faceit_data.csv"],
            "output_files": ["pro_faceit_ids.json", "high_level_faceit_ids.json", "normal_faceit_ids.json"],
            "estimated_time": "15-20 minutes"
        },
        {
            "num": 4,
            "script": "feature_extraction.py",
            "title": "EXTRACT PLAYER FEATURES",
            "description": "Fetching detailed statistics from FACEIT API",
            "required_files": ["pro_faceit_ids.json", "high_level_faceit_ids.json", "normal_faceit_ids.json"],
            "output_files": ["faceit_players_dataset.csv"],
            "estimated_time": "30-60 minutes ⏰ (API rate limited)"
        },
        {
            "num": 5,
            "script": "preprocessing.py",
            "title": "PREPROCESS DATA",
            "description": "Feature engineering and data splitting",
            "required_files": ["faceit_players_dataset.csv"],
            "output_files": ["data/X_train.npy", "data/y_train.npy"],
            "estimated_time": "2-3 minutes"
        },
        {
            "num": 6,
            "script": "model_training.py",
            "title": "TRAIN MODELS",
            "description": "Training and comparing 4 ML models",
            "required_files": ["data/X_train.npy"],
            "output_files": ["models/xgboost.pkl", "models/best_model.txt"],
            "estimated_time": "5-10 minutes"
        },
        {
            "num": 7,
            "script": "calculate_class_stats.py",
            "title": "CALCULATE CLASS STATISTICS",
            "description": "Computing statistics for analysis",
            "required_files": ["faceit_players_dataset.csv"],
            "output_files": ["data/class_statistics.json"],
            "estimated_time": "1 minute"
        }
    ]
    
    # Run each step
    total_start = time.time()
    
    for step in pipeline_steps:
        print_step(step["num"], step["title"])
        print_info(f"⏱️  Estimated time: {step['estimated_time']}")
        
        # Check required files
        if step["required_files"]:
            print_info("Checking required files...")
            for req_file in step["required_files"]:
                if not check_file_exists(req_file):
                    print_error(f"\n⚠️  Required file missing: {req_file}")
                    print_info("Please run previous steps first or check if files exist.")
                    return False
        
        # Run the script
        success = run_script(step["script"], step["description"])
        
        if not success:
            print_error(f"\n⚠️  Pipeline failed at step {step['num']}")
            return False
        
        # Verify output files
        print_info("\nVerifying output files...")
        for output_file in step["output_files"]:
            if not check_file_exists(output_file):
                print_warning(f"Expected output file not found: {output_file}")
    
    # Pipeline complete
    total_time = time.time() - total_start
    
    print_header("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print_success(f"Total time: {total_time/60:.1f} minutes")
    print_info(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "="*70)
    print(f"{Colors.BOLD}📊 NEXT STEPS:{Colors.END}")
    print("="*70)
    print(f"\n{Colors.GREEN}1. Run the web application:{Colors.END}")
    print(f"   {Colors.CYAN}python app.py{Colors.END}")
    print(f"\n{Colors.GREEN}2. Open your browser:{Colors.END}")
    print(f"   {Colors.CYAN}http://localhost:5000{Colors.END}")
    print(f"\n{Colors.GREEN}3. Test with FACEIT nicknames:{Colors.END}")
    print(f"   {Colors.CYAN}Try: s1mple, ZywOo, NiKo{Colors.END}")
    print("\n" + "="*70 + "\n")
    
    return True

def run_single_step(step_number):
    """Run a single step of the pipeline"""
    steps = {
        1: ("pro_players.py", "Collect Pro Players"),
        2: ("data_collection.py", "Collect Player IDs"),
        3: ("feature_extraction.py", "Extract Features"),
        4: ("preprocessing.py", "Preprocess Data"),
        5: ("model_training.py", "Train Models"),
        6: ("calculate_class_stats.py", "Calculate Statistics"),
        7: ("app.py", "Run Web Application")
    }
    
    if step_number not in steps:
        print_error(f"Invalid step number: {step_number}")
        return False
    
    script, description = steps[step_number]
    print_header(f"RUNNING STEP {step_number}: {description.upper()}")
    
    if step_number == 7:
        print_info("Starting web application...")
        print_info("Press Ctrl+C to stop the server")
        subprocess.run([sys.executable, script])
    else:
        return run_script(script, description)

def show_menu():
    """Display interactive menu"""
    print_header("🎮 FACEIT CS2 PLAYER CLASSIFIER - PIPELINE RUNNER")
    
    print(f"\n{Colors.BOLD}Choose an option:{Colors.END}\n")
    print(f"{Colors.CYAN}1.{Colors.END} Run FULL pipeline (all steps)")
    print(f"{Colors.CYAN}2.{Colors.END} Run individual step")
    print(f"{Colors.CYAN}3.{Colors.END} Check system status")
    print(f"{Colors.CYAN}4.{Colors.END} Run web application only")
    print(f"{Colors.CYAN}5.{Colors.END} Exit")
    
    choice = input(f"\n{Colors.BOLD}Enter your choice (1-5): {Colors.END}").strip()
    return choice

def check_system_status():
    """Check which steps have been completed"""
    print_header("📊 SYSTEM STATUS CHECK")
    
    files_to_check = {
        "Pro Players Data": "pro_faceit_data.csv",
        "Pro Player IDs": "pro_faceit_ids.json",
        "High-Level Player IDs": "high_level_faceit_ids.json",
        "Normal Player IDs": "normal_faceit_ids.json",
        "Feature Dataset": "faceit_players_dataset.csv",
        "Training Data": "data/X_train.npy",
        "Trained Models": "models/xgboost.pkl",
        "Class Statistics": "data/class_statistics.json"
    }
    
    print(f"\n{Colors.BOLD}File Status:{Colors.END}\n")
    all_exist = True
    for name, filepath in files_to_check.items():
        exists = os.path.exists(filepath)
        if exists:
            size = os.path.getsize(filepath)
            size_mb = size / (1024 * 1024)
            print_success(f"{name}: Found ({size_mb:.2f} MB)")
        else:
            print_warning(f"{name}: Missing")
            all_exist = False
    
    if all_exist:
        print_success("\n✅ All required files exist! You can run the web app.")
    else:
        print_warning("\n⚠️  Some files are missing. Run the full pipeline.")

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        # Command line mode
        arg = sys.argv[1]
        if arg == "full":
            run_full_pipeline()
        elif arg.isdigit():
            run_single_step(int(arg))
        elif arg == "status":
            check_system_status()
        elif arg == "app":
            run_single_step(7)
        else:
            print_error(f"Unknown argument: {arg}")
            print_info("Usage: python run_pipeline.py [full|1-7|status|app]")
    else:
        # Interactive mode
        while True:
            choice = show_menu()
            
            if choice == "1":
                run_full_pipeline()
                break
            elif choice == "2":
                step = input(f"\n{Colors.BOLD}Enter step number (1-7): {Colors.END}").strip()
                if step.isdigit():
                    run_single_step(int(step))
                else:
                    print_error("Invalid step number")
            elif choice == "3":
                check_system_status()
                input(f"\n{Colors.BOLD}Press Enter to continue...{Colors.END}")
            elif choice == "4":
                run_single_step(7)
                break
            elif choice == "5":
                print_info("\nGoodbye! 👋\n")
                break
            else:
                print_error("Invalid choice. Please try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_warning("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(0)