# 🎮 FACEIT CS2 Player Classification System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

An AI-powered machine learning system that classifies Counter-Strike 2 players on FACEIT into three skill categories (PRO, HIGH-LEVEL, NORMAL) and provides personalized performance analysis and improvement recommendations.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Dataset](#dataset)
- [Technologies](#technologies)
- [Future Improvements](#future-improvements)

## 🎯 Overview

This project demonstrates a complete end-to-end machine learning pipeline:

1. **Data Collection**: Scraping professional player data from HLTV and collecting ranked players from FACEIT API
2. **Feature Engineering**: Extracting 30+ statistical features from player performance
3. **Model Training**: Comparing 4 ML models (Logistic Regression, Random Forest, XGBoost, Neural Network)
4. **Deployment**: Flask web application with interactive UI for real-time predictions

### Classification Categories

- **PRO**: Professional players from competitive teams (HLTV database)
- **HIGH-LEVEL**: Elite players with ELO ≥ 1751 (FACEIT Level 9-10) and FPL participants
- **NORMAL**: Standard players with ELO 500-1350 (FACEIT Level 2-6)

*Note: Level 7-8 players (ELO 1351-1750) are excluded to create clearer class separation.*

## ✨ Features

### Player Analysis
- ✅ Real-time player classification with confidence scores
- ✅ Detailed performance statistics (K/D, Win Rate, Headshot %, etc.)
- ✅ Strengths and weaknesses identification
- ✅ Personalized improvement recommendations

### Technical Features
- ✅ RESTful API with Flask backend
- ✅ Responsive web interface
- ✅ 30+ engineered features for prediction
- ✅ Model comparison framework
- ✅ Comprehensive data preprocessing pipeline

## 📁 Project Structure

```
faceit-cs2-classifier/
│
├── data/                          # Processed datasets
│   ├── X_train.npy
│   ├── X_val.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   ├── y_val.npy
│   ├── y_test.npy
│   ├── feature_names.json
│   └── class_statistics.json
│
├── models/                        # Trained models
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── neural_network.h5
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── best_model.txt
│
├── templates/                     # HTML templates
│   └── index.html
│
├── results/                       # Training results
│   └── model_comparison.csv
│
├── pro_players.py                 # Scrape pro players from HLTV
├── data_collection.py             # Collect player IDs by class
├── feature_extraction.py          # Extract features from FACEIT API
├── preprocessing.py               # Data preprocessing pipeline
├── model_training.py              # Train and compare models
├── calculate_class_stats.py       # Calculate class statistics
├── app.py                         # Flask web application
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- FACEIT API key (free from https://developers.faceit.com/)

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/faceit-cs2-classifier.git
cd faceit-cs2-classifier
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure API Key**
   
   Edit `feature_extraction.py` and `app.py`:
   ```python
   FACEIT_API_KEY = "your-api-key-here"
   ```

## 📊 Usage

### Step 1: Collect Pro Players
```bash
python pro_players.py
```
This scrapes professional player data from HLTV and saves to `pro_faceit_data.csv`.

### Step 2: Collect Player IDs
```bash
python data_collection.py
```
This collects:
- PRO player IDs from HLTV data
- HIGH-LEVEL player IDs from FPL and top MM rankings
- NORMAL player IDs from mid-tier MM rankings

Output files:
- `pro_faceit_ids.json`
- `high_level_faceit_ids.json`
- `normal_faceit_ids.json`

### Step 3: Extract Features
```bash
python feature_extraction.py
```
Fetches detailed statistics for each player from FACEIT API and creates `faceit_players_dataset.csv`.

**Note**: This step takes 30-60 minutes depending on dataset size due to API rate limits.

### Step 4: Preprocess Data
```bash
python preprocessing.py
```
Performs:
- Missing value handling
- Feature engineering (10 derived features)
- Feature scaling (StandardScaler)
- Train/Val/Test split (70/15/15)

### Step 5: Train Models
```bash
python model_training.py
```
Trains and evaluates 4 models:
- Logistic Regression (baseline)
- Random Forest
- XGBoost
- Neural Network

Saves best model automatically.

### Step 6: Calculate Class Statistics
```bash
python calculate_class_stats.py
```
Computes mean/std statistics for each class to enable strength/weakness analysis.

### Step 7: Run Web Application
```bash
python app.py
```
Access the application at `http://localhost:5000`

## 📈 Model Performance

Expected performance on test set:

| Model | Accuracy | Notes |
|-------|----------|-------|
| Logistic Regression | ~75-80% | Fast baseline |
| Random Forest | ~85-90% | Good interpretability |
| XGBoost | ~88-93% | **Best performance** |
| Neural Network | ~86-91% | Requires more data |

*Actual results may vary based on dataset quality and size.*

### Feature Importance (Top 10)

1. `faceit_elo` - Current ELO rating
2. `skill_level` - FACEIT level (1-10)
3. `kd_ratio` - Kill/Death ratio
4. `win_rate` - Overall win percentage
5. `avg_headshots` - Headshot accuracy
6. `kr_ratio` - Kills per round
7. `impact_score` - Derived impact metric
8. `headshot_skill` - Composite headshot metric
9. `win_contribution` - Win rate × K/D
10. `matches` - Experience level

## 📦 Dataset

### Data Sources

1. **PRO Players**: HLTV.org stats page
   - ~578 professional players
   - Active team members only

2. **HIGH-LEVEL Players**: 
   - FACEIT Pro League (FPL) participants
   - Top 2000 MM ranking (ELO ≥ 1751)
   - Target: ~700 players

3. **NORMAL Players**:
   - MM ranking positions 50k-200k
   - ELO range: 500-1350
   - Target: ~700 players

### Features (30+)

**Basic Statistics:**
- ELO, Skill Level, Matches Played
- K/D Ratio, K/R Ratio, Win Rate
- Average Kills, Deaths, Assists
- Headshot %, MVPs

**Advanced Metrics:**
- Triple/Quadro/Penta Kills
- Win Streaks (current & longest)
- Recent form (last 20 matches)

**Engineered Features:**
- Kill efficiency, Survival rate
- Impact score, Win contribution
- Special kills rate, Consistency
- Headshot skill, Experience level

## 🛠️ Technologies

### Machine Learning
- **scikit-learn**: Preprocessing, classical ML models
- **XGBoost**: Gradient boosting classifier
- **TensorFlow/Keras**: Neural network implementation
- **NumPy/Pandas**: Data manipulation

### Web Development
- **Flask**: Backend API framework
- **HTML/CSS/JavaScript**: Frontend interface
- **RESTful API**: Clean architecture

### Data Collection
- **requests**: FACEIT API interactions
- **BeautifulSoup4**: HTML parsing
- **cloudscraper**: Cloudflare bypass for HLTV

## 🔮 Future Improvements

### Technical Enhancements
- [ ] Implement SHAP values for model explainability
- [ ] Add hyperparameter tuning (GridSearchCV/Optuna)
- [ ] Create ensemble model combining top performers
- [ ] Implement caching for faster predictions

### Feature Additions
- [ ] Historical performance tracking
- [ ] Compare with similar players
- [ ] Map-specific statistics
- [ ] Weapon preference analysis
- [ ] Role classification (Entry, AWPer, Support, IGL)

### Application Improvements
- [ ] User authentication and profile saving
- [ ] Player comparison tool
- [ ] Export reports as PDF
- [ ] Multi-language support
- [ ] Mobile app version

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

Your Name - [LinkedIn Profile](https://linkedin.com/in/yourprofile)

Project Link: [https://github.com/yourusername/faceit-cs2-classifier](https://github.com/yourusername/faceit-cs2-classifier)

## 🙏 Acknowledgments

- FACEIT for providing comprehensive API
- HLTV.org for professional player statistics
- scikit-learn and TensorFlow communities
- Flask documentation and examples

---

**Built with ❤️ for CS2 players looking to improve their game**