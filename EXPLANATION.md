# Complete Technical Explanation: Climate Forecasting System with RL-Based Model Selection

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Data Pipeline](#2-data-pipeline)
3. [Forecasting Models](#3-forecasting-models)
4. [RL Agent Architecture](#4-rl-agent-architecture)
5. [RAG System](#5-rag-system)
6. [Implementation Details](#6-implementation-details)
7. [Design Decisions & Rationale](#7-design-decisions--rationale)
8. [Performance Analysis](#8-performance-analysis)

---

## 1. System Overview

### 1.1 Problem Statement

Climate change analysis for Algeria requires:
- Accurate long-term forecasts (until 2040) for temperature and ET0
- Multiple forecasting models with different strengths
- Intelligent model selection that adapts to seasonal patterns
- User-friendly query interface for insights

### 1.2 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA PIPELINE                                │
│  Open-Meteo API → Preprocessing → Monthly Aggregation           │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│                  FORECASTING MODELS                              │
│  SARIMA │ LSTM │ Ridge │ Prophet → Individual Forecasts         │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│              RL AGENT (Model Selection)                          │
│  Trained on 2019-2023 → Selects best model per month            │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│                   RAG SYSTEM                                     │
│  stats_db.json → ChromaDB → LLM → User Queries                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Technology Stack

**Core:**
- Python 3.11
- Pandas (data manipulation)
- NumPy (numerical computing)

**Forecasting:**
- pmdarima (SARIMA)
- TensorFlow/Keras (LSTM)
- scikit-learn (Ridge, preprocessing)
- Prophet (Facebook's time series library)

**RL:**
- Custom Q-learning implementation
- pickle (Q-table persistence)

**RAG:**
- ChromaDB (vector database)
- Google Gemini API (LLM)
- Streamlit (UI)

---

## 2. Data Pipeline

### 2.1 Data Source

**API:** Open-Meteo Historical Weather API
- Endpoint: `https://archive-api.open-meteo.com/v1/archive`
- Location: Algiers, Algeria (36.7538°N, 3.0588°E)
- Period: 2002-01-01 to 2023-08-01
- Frequency: Daily data aggregated to monthly

**Variables Retrieved:**
- `temperature_2m_mean` (°C)
- `temperature_2m_max` (°C)
- `temperature_2m_min` (°C)
- `precipitation_sum` (mm)
- `et0_fao_evapotranspiration` (mm)
- `shortwave_radiation_sum` (MJ/m²)
- `windspeed_10m_max` (km/h)

### 2.2 Preprocessing Pipeline

**File:** `Preprocessed_dataset/algiers_monthly_processed_v2.csv`

**Steps:**

1. **Daily to Monthly Aggregation:**
```python
# Group by year-month
monthly_data = daily_data.resample('MS').agg({
    'temperature_2m_mean': 'mean',
    'temperature_2m_max': 'mean',
    'temperature_2m_min': 'mean',
    'precipitation_sum': 'sum',
    'et0_fao_evapotranspiration': 'sum',
    'shortwave_radiation_sum': 'sum',
    'windspeed_10m_max': 'max'
})
```

2. **Feature Engineering:**
```python
# Temperature anomaly (deviation from mean)
df['temp_anomaly'] = df['temperature_2m_mean'] - df['temperature_2m_mean'].mean()

# Water balance (precipitation - evapotranspiration)
df['water_balance'] = df['precipitation_sum'] - df['et0_fao_evapotranspiration']

# Cumulative water balance
df['cumulative_water_balance'] = df['water_balance'].cumsum()

# Aridity index (ET0 / Precipitation)
df['aridity_index'] = df['et0_fao_evapotranspiration'] / df['precipitation_sum']

# Rolling averages (12-month)
df['temp_rolling_12m'] = df['temperature_2m_mean'].rolling(12).mean()
df['precip_rolling_12m'] = df['precipitation_sum'].rolling(12).sum()

# Drought indices (SPI, SPEI)
df['SPI_12_z'] = (df['precip_rolling_12m'] - df['precip_rolling_12m'].mean()) / df['precip_rolling_12m'].std()
df['SPEI_12_z'] = (df['water_balance'].rolling(12).sum() - df['water_balance'].rolling(12).sum().mean()) / df['water_balance'].rolling(12).sum().std()
```

**Output:** 260 monthly samples with 15 features

### 2.3 Data Splits

**Historical Data (2002-2023):**
- Used for model training and validation

**Backcast Split (for RL training):**
- Training: 2002-2018 (204 months)
- Testing: 2019-2023 (56 months)

**Forecast Horizon:**
- 2024-01-01 to 2040-12-01 (204 months)

**Critical Design Decision:**
We use 2019-2023 as the RL training period because:
1. Recent enough to be representative
2. Long enough (56 months) for meaningful learning
3. Tests model generalization on unseen data

---

## 3. Forecasting Models

### 3.1 SARIMA (Seasonal AutoRegressive Integrated Moving Average)

**Purpose:** Captures linear trends and seasonal patterns

**Implementation:**
```python
from pmdarima import auto_arima

model = auto_arima(
    train_data,
    seasonal=True,
    m=12,  # Monthly seasonality
    max_p=5, max_q=5,  # AR and MA terms
    max_P=3, max_Q=3,  # Seasonal AR and MA terms
    information_criterion='aic',
    stepwise=True
)
```

**How It Works:**
- **AR (p):** Uses past values: `y_t = φ₁y_{t-1} + φ₂y_{t-2} + ...`
- **MA (q):** Uses past errors: `+ θ₁ε_{t-1} + θ₂ε_{t-2} + ...`
- **Seasonal (P, Q):** Same as AR/MA but for seasonal lags (12, 24, 36 months)
- **Differencing (d, D):** Makes data stationary

**Strengths:**
- Excellent for linear trends
- Handles seasonality explicitly
- Interpretable coefficients

**Weaknesses:**
- Assumes linear relationships
- Sensitive to outliers

**Performance (2019-2023):**
- Temperature: 1.106°C MAE
- ET0: 17.385 mm MAE

### 3.2 LSTM (Long Short-Term Memory)

**Purpose:** Captures non-linear patterns and long-term dependencies

**Architecture:**
```python
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True, input_shape=(24, 1))),
    Dropout(0.2),
    Bidirectional(LSTM(32, return_sequences=False)),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.1),
    Dense(1)
])
```

**Key Components:**

1. **Bidirectional LSTM:**
   - Processes sequence forward AND backward
   - Captures patterns from both directions
   - 64 units in first layer, 32 in second

2. **Lookback Window:** 24 months
   - Uses 2 years of history for each prediction
   - Balances context vs overfitting

3. **Dropout (0.1-0.2):**
   - Prevents overfitting
   - Different rates for different layers

4. **MinMax Scaling:**
```python
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
# ... train model ...
predictions = scaler.inverse_transform(scaled_predictions)
```

**How It Works:**
- **Cell State:** Long-term memory
- **Hidden State:** Short-term memory
- **Gates:** Control information flow
  - Forget gate: What to remove from memory
  - Input gate: What new information to store
  - Output gate: What to output

**Iterative Forecasting:**
```python
last_sequence = train_data[-24:]  # Last 24 months
predictions = []

for i in range(204):  # Forecast until 2040
    pred = model.predict(last_sequence)
    predictions.append(pred)
    # Roll window: remove oldest, add newest
    last_sequence = np.append(last_sequence[1:], pred)
```

**Strengths:**
- Learns non-linear patterns
- Handles complex seasonality
- Robust to noise

**Weaknesses:**
- "Black box" (not interpretable)
- Requires more data
- Computationally expensive

**Performance (2019-2023):**
- Temperature: 1.084°C MAE ⭐ **Best**
- ET0: 15.432 mm MAE ⭐ **Best**

### 3.3 Ridge Regression (Polynomial)

**Purpose:** Captures trend with regularization

**Implementation:**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),  # x, x²
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))
])
```

**How It Works:**

1. **Polynomial Features:**
```python
# Input: [1, 2, 3, ...]  (time index)
# Output: [1, 1², 2, 2², 3, 3², ...]
```

2. **Ridge Penalty:**
```python
# Minimize: MSE + α × Σ(β²)
# α=1.0 provides moderate regularization
```

**Why Polynomial Degree 2?**
- Degree 1: Too simple (linear only)
- Degree 2: Captures quadratic trends
- Degree 3+: Overfits on 260 samples

**Strengths:**
- Fast to train
- Smooth predictions
- Regularization prevents overfitting

**Weaknesses:**
- Cannot capture seasonality
- Fixed functional form
- Poor long-term extrapolation

**Performance (2019-2023):**
- Temperature: 5.174°C MAE ❌
- ET0: 45.061 mm MAE ❌

### 3.4 Prophet

**Purpose:** Additive model with automatic changepoint detection

**Implementation:**
```python
from prophet import Prophet

model = Prophet(
    seasonality_mode='multiplicative',  # Seasonal effect grows with trend
    yearly_seasonality=True,
    daily_seasonality=False,
    weekly_seasonality=False
)

# Add custom monthly seasonality
model.add_seasonality(
    name='monthly',
    period=30.5,  # Average month length
    fourier_order=5  # Complexity of seasonal pattern
)

model.fit(train_df)
```

**How It Works:**

Prophet decomposes time series as:
```
y(t) = g(t) + s(t) + h(t) + ε(t)
```

Where:
- **g(t):** Trend (piecewise linear with changepoints)
- **s(t):** Seasonality (Fourier series)
- **h(t):** Holidays (not used here)
- **ε(t):** Error term

**Fourier Seasonality:**
```python
# 5th order Fourier series:
s(t) = Σ[aₙ·cos(2πnt/365) + bₙ·sin(2πnt/365)], n=1..5
```

**Strengths:**
- Automatic trend changepoints
- Handles missing data
- Uncertainty intervals (yhat_lower, yhat_upper)

**Weaknesses:**
- Can overfit short series
- Assumes additive/multiplicative form
- Not ideal for complex patterns

**Performance (2019-2023):**
- Temperature: 4.126°C MAE ❌
- ET0: 48.739 mm MAE ❌

**Why Prophet Underperformed:**
- 260 samples too small for robust changepoint detection
- Linear trend assumption doesn't match climate complexity
- LSTM/SARIMA better suited for this dataset

---

## 4. RL Agent Architecture

### 4.1 Q-Learning Fundamentals

**Core Concept:** Learn action-value function Q(s, a)

**Q-Table Structure:**
```
        SARIMA  LSTM    Ridge   Prophet
Jan     -10.06  -11.03  -17.00  -11.86
Feb     -9.68   -11.69  -16.77  -11.80
Mar     -10.20  -10.93  -14.59  -14.76
...
Dec     -11.30  -10.86  -13.82  -15.44
```

- **Rows (States):** 12 months (January-December)
- **Columns (Actions):** 4 models
- **Values:** Expected cumulative reward

### 4.2 Training Algorithm

**File:** `RL Agent/pretrain_agent.py`

**Key Parameters:**
```python
agent = RLAgent(
    n_actions=4,
    lr=0.5,           # Learning rate
    gamma=0.9,        # Discount factor
    epsilon_start=1.0, # Initial exploration
    epsilon_end=0.01,  # Final exploration
    epsilon_decay=0.995
)
```

**Training Loop (1000 episodes):**
```python
for episode in range(1000):
    for month_idx in range(56):  # 2019-2023
        # 1. Select action (ε-greedy)
        if random() < epsilon:
            action = random_choice([0, 1, 2, 3])
        else:
            action = argmax(Q[state, :])
        
        # 2. Get predictions for BOTH variables
        temp_pred = temp_df[model_names[action]][month_idx]
        et0_pred = et0_df[model_names[action]][month_idx]
        
        # 3. Calculate normalized combined error
        temp_error = |actual_temp - temp_pred| / best_temp_mae
        et0_error = |actual_et0 - et0_pred| / best_et0_mae
        combined_error = (temp_error + et0_error) / 2
        
        # 4. Reward = negative error
        reward = -combined_error
        
        # 5. Q-Learning update
        Q[state, action] += lr * (
            reward + gamma * max(Q[next_state, :]) - Q[state, action]
        )
        
        # 6. Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
```

**Why This Update Rule?**
```
Q_new = Q_old + α[R + γ·max(Q_next) - Q_old]
        ︸︷︷︸   ︸︷︷︸   ︸︷︷︸    ︸︷︷︸
        old     reward  future   current
       value            value    value
```

- **R:** Immediate reward (how good was this choice?)
- **γ·max(Q_next):** Future rewards (what's the best we can do next?)
- **Difference:** How much to adjust our estimate

### 4.3 Dual-Variable Reward System

**Critical Innovation:** Train on BOTH temperature and ET0

**Why?**
Single-variable training had fatal flaw:
- Ridge: Good for temperature May (10.58°C vs 10.81°C LSTM)
- Ridge: Terrible for ET0 May (158.16 mm vs 146.92 mm LSTM)
- Result: Ridge selected 8.3% despite being 3.85x worse overall!

**Solution:**
```python
# Normalize by best performance
temp_best = 1.084  # LSTM's MAE on temperature
et0_best = 15.432  # LSTM's MAE on ET0

# For each prediction:
temp_normalized_error = |actual - pred| / temp_best
et0_normalized_error = |actual - pred| / et0_best

# Combined reward (equal weight)
combined_reward = -(temp_normalized_error + et0_normalized_error) / 2
```

**Effect:**
- Model must be good at BOTH variables
- Bad performance on either → low reward
- LSTM: 1.00x temp, 1.00x ET0 → Best overall
- SARIMA: 1.02x temp, 1.13x ET0 → Second best
- Ridge: 4.77x temp, 2.92x ET0 → Eliminated
- Prophet: 3.81x temp, 3.16x ET0 → Eliminated

### 4.4 Learned Preferences

**Final Q-Table (After 1000 Episodes):**

Best model by month:
- Jan: LSTM (-8.76 best)
- Feb-Apr: SARIMA
- May-Sep: LSTM  
- Oct: SARIMA
- Nov-Dec: LSTM

**Pattern:**
- **LSTM dominates** (8/12 months = 66.7%)
- **SARIMA for transitions** (4/12 months = 33.3%)
- **Ridge/Prophet never selected** (eliminated)

**Why This Pattern?**
- **LSTM:** Best at capturing complex patterns, extreme months
- **SARIMA:** Better for stable transition periods (Feb-Apr, Oct)
- **Ridge:** Too simplistic, no seasonality
- **Prophet:** Overfits on small dataset

### 4.5 Deployment

**File:** `RL Agent/rl_forecast_unified.py`

**Process:**
```python
# 1. Load pre-trained Q-table
agent.load_model("pretrained_q_table.pkl")

# 2. Load all 4 model forecasts (2024-2040)
sarima_forecast = pd.read_csv("Predictions/sarima_forecast_2040.csv")
lstm_forecast = pd.read_csv("Predictions/lstm_forecast_2040.csv")
ridge_forecast = pd.read_csv("Predictions/ridge_forecast_2040.csv")
prophet_forecast = pd.read_csv("Predictions/prophet_forecast_2040.csv")

# 3. For each month (2024-2040):
for date in forecast_dates:
    month = date.month - 1  # 0-indexed
    
    # Select best model for this month
    action = np.argmax(agent.q_table[month, :])
    selected_model = model_names[action]
    
    # Get prediction from selected model
    if action == 0:
        prediction = sarima_forecast.loc[date]
    elif action == 1:
        prediction = lstm_forecast.loc[date]
    # ... etc
    
    # Store result
    results.append({
        'Date': date,
        'RL_Best_Forecast': prediction,
        'Model_Used': selected_model,
        'SARIMA': sarima_forecast.loc[date],
        'LSTM': lstm_forecast.loc[date],
        'Ridge': ridge_forecast.loc[date],
        'Prophet': prophet_forecast.loc[date]
    })

# 4. Save final forecast
results.to_csv("final_rl_temperature_forecast_2040.csv")
```

**Output:** 204 predictions with model attribution

---

## 5. RAG System (Retrieval-Augmented Generation)

### 5.1 Architecture

**Purpose:** Natural language interface for climate insights

**Components:**
```
User Query
    ↓
┌─────────────────────────────────────┐
│ 1. stats_db.json (Structured Data)  │  ← Forecast statistics
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ 2. ChromaDB (Vector Database)       │  ← Semantic search
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ 3. Gemini LLM (Google AI)           │  ← Answer generation
└─────────────┬───────────────────────┘
              ↓
         Natural Language Answer
```

### 5.2 stats_db.json Generation

**File:** `src/generate_stats_db.py`

**Process:**
```python
def generate_stats_db():
    # 1. Load trend analysis (if exists)
    trends = load_trends()  # From Results/trends.json
    
    # 2. Load RL forecasts for BOTH variables
    temp_forecast = pd.read_csv('RL Agent/final_rl_temperature_forecast_2040.csv')
    et0_forecast = pd.read_csv('RL Agent/final_rl_et0_forecast_2040.csv')
    
    # 3. Extract statistics
    forecasts = {
        'RL_Agent_Temperature': {
            'model': 'RL Agent - Temperature (Dual-Variable Trained)',
            'variable': 'temperature_2m_mean',
            'unit': '°C',
            'forecast_period': '2024-01-01 to 2040-12-01',
            'training_method': 'Dual-variable Q-learning',
            'summary_statistics': {
                'RL_Best_Forecast': {
                    'count': 204,
                    'mean': df['RL_Best_Forecast'].mean(),
                    'std': df['RL_Best_Forecast'].std(),
                    'min': df['RL_Best_Forecast'].min(),
                    'max': df['RL_Best_Forecast'].max(),
                    '25%': df['RL_Best_Forecast'].quantile(0.25),
                    '50%': df['RL_Best_Forecast'].median(),
                    '75%': df['RL_Best_Forecast'].quantile(0.75)
                },
                'Model_Selection': {
                    'LSTM': 136,
                    'SARIMA': 68
                }
            },
            'last_5_years': df.tail(60).to_dict('records'),
            'full_forecast': df.to_dict('records')
        },
        'RL_Agent_ET0': {
            # Same structure for ET0
        }
    }
    
    # 4. Create structured database
    stats_db = {
        'metadata': {
            'generated_at': timestamp,
            'description': 'Climate forecasts for Algeria',
            'data_sources': ['RL Agent forecasts']
        },
        'trends': trends,
        'forecasts': forecasts,
        'summary': {
            'num_forecast_models': 2,
            'forecast_models': ['RL_Agent_Temperature', 'RL_Agent_ET0']
        }
    }
    
    # 5. Save as JSON
    with open('Results/stats_db.json', 'w') as f:
        json.dump(stats_db, f, indent=2)
```

**Output Size:** ~1.5 MB JSON file with all forecast data

### 5.3 ChromaDB Integration

**File:** `src/rag.py`

**Initialization:**
```python
import chromadb
from chromadb.config import Settings

# Persistent storage
client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

# Create or get collection
collection = client.get_or_create_collection(
    name="climate_stats",
    metadata={"hnsw:space": "cosine"}  # Cosine similarity
)
```

**Document Preparation:**
```python
def load_stats_to_chromadb():
    # 1. Load stats_db.json
    with open('Results/stats_db.json') as f:
        stats_db = json.load(f)
    
    # 2. Create documents for each forecast model
    documents = []
    metadatas = []
    ids = []
    
    for model_name, model_data in stats_db['forecasts'].items():
        # Create rich text description
        doc_text = f"""
        Model: {model_data['model']}
        Variable: {model_data['variable']}
        Unit: {model_data['unit']}
        Forecast Period: {model_data['forecast_period']}
        Training Method: {model_data['training_method']}
        
        Statistics:
        - Mean: {model_data['summary_statistics']['RL_Best_Forecast']['mean']:.2f}
        - Std Dev: {model_data['summary_statistics']['RL_Best_Forecast']['std']:.2f}
        - Min: {model_data['summary_statistics']['RL_Best_Forecast']['min']:.2f}
        - Max: {model_data['summary_statistics']['RL_Best_Forecast']['max']:.2f}
        
        Model Selection:
        {json.dumps(model_data['summary_statistics']['Model_Selection'], indent=2)}
        
        Recent Forecast (Last 5 years):
        {json.dumps(model_data['last_5_years'], indent=2)}
        """
        
        documents.append(doc_text)
        metadatas.append({
            'type': 'forecast',
            'model': model_name,
            'variable': model_data['variable']
        })
        ids.append(f"forecast_{model_name}")
    
    # 3. Add to ChromaDB
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
```

**How ChromaDB Works:**

1. **Embedding Creation:**
   - Default: all-MiniLM-L6-v2 (sentence-transformers)
   - Converts text → 384-dimensional vector
   ```python
   "temperature forecast" → [0.23, -0.15, 0.89, ...]
   ```

2. **Vector Storage:**
   - HNSW index (Hierarchical Navigable Small World)
   - Fast approximate nearest neighbor search
   - O(log n) query time

3. **Similarity Search:**
   ```python
   query = "What will temperature be in 2040?"
   # Query embedding: [0.21, -0.18, 0.91, ...]
   
   # Cosine similarity with all documents
   similarity = dot(query_vec, doc_vec) / (norm(query_vec) * norm(doc_vec))
   
   # Return top k most similar
   results = collection.query(
       query_texts=[query],
       n_results=3
   )
   ```

### 5.4 LLM Integration (Google Gemini)

**File:** `src/rag.py` (continued)

**Query Process:**
```python
def rag_query(user_question: str) -> str:
    # 1. Retrieve relevant documents from ChromaDB
    results = collection.query(
        query_texts=[user_question],
        n_results=5  # Top 5 most relevant
    )
    
    # 2. Combine retrieved context
    context = "\n\n".join(results['documents'][0])
    
    # 3. Build prompt for LLM
    prompt = f"""
    You are a climate analysis assistant for Algeria.
    
    User Question: {user_question}
    
    Relevant Climate Data:
    {context}
    
    Instructions:
    - Answer based ONLY on the provided data
    - Be specific with numbers and dates
    - If data is insufficient, say so
    - Format: Clear, concise, scientific
    
    Answer:
    """
    
    # 4. Call Gemini API
    response = genai.generate_text(
        model='gemini-pro',
        prompt=prompt,
        temperature=0.3,  # Low = more factual
        max_output_tokens=500
    )
    
    return response.text
```

**Why Gemini (not local LLM)?**
- Better Algerian climate understanding
- More accurate numerical reasoning  
- Faster response time
- No GPU required

### 5.5 Streamlit UI

**File:** `src/app.py`

**Key Components:**

1. **Session State Management:**
```python
if 'messages' not in st.session_state:
    st.session_state.messages = []  # Chat history

if 'rag_system' not in st.session_state:
    st.session_state.rag_system = RAGSystem()  # Singleton
```

2. **Database Reload:**
```python
if st.sidebar.button("🔄 Reload DB"):
    # Clear chromaDB
    st.session_state.rag_system.collection.delete()
    
    # Reload from stats_db.json
    st.session_state.rag_system.load_stats()
    
    st.success("Database reloaded!")
```

3. **Chat Interface:**
```python
# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# User input
if prompt := st.chat_input("Ask about climate..."):
    # Add user message
    st.session_state.messages.append({
        'role': 'user',
        'content': prompt
    })
    
    # Get RAG response
    with st.spinner("Analyzing..."):
        response = st.session_state.rag_system.query(prompt)
    
    # Add assistant message  
    st.session_state.messages.append({
        'role': 'assistant',
        'content': response
    })
    
    # Rerun to display
    st.rerun()
```

**Example Queries:**

1. **"What will temperature be in 2040?"**
   - ChromaDB finds: RL_Agent_Temperature forecast data
   - LLM extracts: Mean, range, seasonal pattern
   - Response: "Based on RL agent forecasts, average temperature in 2040 is projected to be 18.5°C, ranging from 11°C (winter) to 27°C (summer)."

2. **"Which model does the RL agent use most?"**
   - ChromaDB finds: Model selection statistics
   - LLM interprets: LSTM 66.7%, SARIMA 33.3%
   - Response: "The RL agent predominantly selects LSTM (136 out of 204 months, 66.7%) for its superior accuracy..."

3. **"Why not use Prophet?"**
   - ChromaDB finds: Training method description
   - LLM explains: Dual-variable training eliminated poor performers
   - Response: "Prophet was excluded by the RL agent due to poor combined performance (3.48x worse than LSTM)..."

---

## 6. Implementation Details

### 6.1 File Structure

```
Project-MLA/
├── Dataset/
│   └── Algiers_Weather_Data.csv          # Raw daily data
├── Preprocessed_dataset/
│   └── algiers_monthly_processed_v2.csv  # Monthly processed
├── Predictions/
│   ├── sarima_forecast_2040.csv          # SARIMA predictions
│   ├── lstm_forecast_2040.csv            # LSTM predictions
│   ├── ridge_forecast_2040.csv           # Ridge predictions
│   └── prophet_forecast_2040.csv         # Prophet predictions
├── RL Agent/
│   ├── agent.py                          # RLAgent class
│   ├── pretrain_agent.py                 # Dual-variable training
│   ├── rl_forecast_unified.py            # Generate final forecasts
│   ├── pretrained_q_table.pkl            # Saved Q-table
│   ├── historical_backcasts_temperature_2019_2023_real.csv
│   ├── historical_backcasts_et0_2019_2023_real.csv
│   ├── final_rl_temperature_forecast_2040.csv
│   └── final_rl_et0_forecast_2040.csv
├── Forecasting_Models/
│   ├── unified_forecast.ipynb            # Main forecasting notebook
│   └── backcast_generator.ipynb          # Generate RL training data
├── src/
│   ├── app.py                            # Streamlit UI
│   ├── rag.py                            # RAG system
│   └── generate_stats_db.py              # Create stats_db.json
├── Results/
│   └── stats_db.json                     # Aggregated statistics
├── chroma_db/                            # ChromaDB storage
└── requirements.txt                      # Python dependencies
```

### 6.2 RLAgent Class

**File:** `RL Agent/agent.py`

```python
class RLAgent:
    def __init__(self, n_actions=4, lr=0.5, gamma=0.9, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        """
        Q-Learning agent for model selection
        
        Args:
            n_actions: Number of models (4: SARIMA, LSTM, Ridge, Prophet)
            lr: Learning rate (how much to update Q-values)
            gamma: Discount factor (future reward importance)
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Exploration decay per episode
        """
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q-table: [12 states x 4 actions]
        self.q_table = np.zeros((12, n_actions))
    
    def decision_policy(self, state):
        """
        ε-greedy action selection
        
        Args:
            state: Month index (0-11)
        
        Returns:
            action: Model index (0-3)
        """
        if np.random.uniform(0, 1) < self.epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: best known action
            return np.argmax(self.q_table[state, :])
    
    def update(self, state, action, reward, next_state):
        """
        Q-learning update rule
        
        Q(s,a) ← Q(s,a) + α[R + γ·max Q(s',a') - Q(s,a)]
        """
        predict = self.q_table[state, action]
        target = reward + self.gamma * np.max(self.q_table[next_state, :])
        self.q_table[state, action] += self.lr * (target - predict)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filename):
        """Save Q-table to disk"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon
            }, f)
    
    def load_model(self, filename):
        """Load Q-table from disk"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.epsilon = data['epsilon']
```

### 6.3 Critical Code Patterns

**1. Handling Missing Data:**
```python
# In backcast generator
try:
    data = pd.read_csv('historical_backcasts_temperature_2019_2023_real.csv')
except FileNotFoundError:
    # Fallback to old file or error
    print("⚠️ Real backcasts not found. Run backcast_generator.ipynb first.")
    return None
```

**2. Data Validation:**
```python
# Ensure date continuity
dates = pd.date_range('2024-01-01', '2040-12-01', freq='MS')
assert len(forecast) == len(dates), "Forecast length mismatch!"

# Check for NaN
assert not forecast.isna().any(), "Forecast contains NaN values!"
```

**3. Consistent Column Names:**
```python
# All forecast CSVs have:
# - Date (datetime)
# - Forecast (float)

# RL output has:
# - Date
# - SARIMA, LSTM, Ridge, Prophet (individual forecasts)
# - RL_Best_Forecast (selected)
# - Model_Used (model name)
```

**4. Error Handling in RAG:**
```python
try:
    response = rag_system.query(user_question)
except Exception as e:
    # Graceful degradation
    response = f"⚠️ Error: {str(e)}. Please try rephrasing your question."
    logger.error(f"RAG query failed: {e}")
```

---

## 7. Design Decisions & Rationale

### 7.1 Why Dual-Variable Training?

**Decision:** Train RL agent on combined temperature + ET0 rewards

**Rationale:**
- Single-variable training had critical flaw: Ridge selected despite being 3.85x worse overall
- Climate applications need consistency across variables
- Dual-variable approach ensures robust model selection

**Alternative Considered:**
- Separate Q-tables for each variable → Rejected (more complex, inconsistent selections)

### 7.2 Why 2019-2023 as Training Period?

**Decision:** Use 2019-2023 (56 months) for RL training

**Rationale:**
- Recent enough to be representative of current climate
- Long enough for meaningful statistical learning
- Tests model generalization on unseen data
- Avoids data leakage (not used in model training)

**Alternative Considered:**
- Full 2002-2023 → Rejected (would test on training data, overfitting)
- Only 2020-2023 → Rejected (too short, only 44 months)

### 7.3 Why Q-Learning (Not Deep RL)?

**Decision:** Use tabular Q-learning, not DQN/A3C

**Rationale:**
- State space is small (12 months)
- Action space is small (4 models)
- Tabular Q-learning is:
  - Interpretable (can see exact Q-values)
  - Guaranteed to converge
  - No hyperparameter tuning
  - Fast to train (<1 minute)

**When Would Deep RL Be Better?**
- 100+ states (e.g., daily forecasting)
- Continuous state space (e.g., temperature as state)
- Complex features (weather patterns, pressure, etc.)

### 7.4 Why Not Weighted Ensemble?

**Decision:** Select one model, don't average

**Rationale:**
1. **Interpretability:** Can explain which model was used
2. **Simplicity:** No weight optimization needed
3. **Performance:** Best model often better than average
4. **RL Suitability:** Discrete actions fit Q-learning

**Alternative Considered:**
- RL learns weights, averages predictions → More complex, less interpretable

### 7.5 Why ChromaDB (Not SQL)?

**Decision:** Use vector database, not relational DB

**Rationale:**
- **Semantic Search:** "temperature trend" matches "climate warming trend"
- **Flexible Schema:** JSON documents, no fixed schema
- **Fast Retrieval:** Approximate nearest neighbors
- **No SQL Required:** Simpler queries

**When Would SQL Be Better?**
- Structured queries: "SELECT * WHERE date > 2030"
- Transactions, ACID guarantees
- Complex joins across tables

### 7.6 Why 24-Month Lookback for LSTM?

**Decision:** Use 24-month window (2 years)

**Rationale:**
- **Too short (6-12 months):** Misses annual seasonality
- **24 months:** Captures full seasonal cycle + trend
- **Too long (48+ months):** Overfits, data scarcity (only 260 total)

**Formula:**
```
Max lookback = Total_samples / 5 = 260 / 5 = 52 months
Used lookback = 24 months (conservative)
```

### 7.7 Why Polynomial Degree 2 for Ridge?

**Decision:** Use quadratic features (degree=2)

**Rationale:**
- **Degree 1 (Linear):** Underfits, can't capture acceleration
- **Degree 2 (Quadratic):** Captures trend changes
- **Degree 3+:** Overfits on 260 samples

**Overfitting Test:**
```python
# Features per degree
degree_1_features = 1  # [x]
degree_2_features = 2  # [x, x²]
degree_3_features = 3  # [x, x², x³]

# Rule: samples > 10 × features
260 > 10 × 2 = 20 ✓  # Degree 2 safe
260 > 10 × 3 = 30 ✓  # Degree 3 borderline
```

### 7.8 Why Save Q-Table (Not Retrain)?

**Decision:** Save and load Q-table, don't retrain on every forecast

**Rationale:**
- Training takes 1000 episodes (~1 minute)
- Q-table is deterministic after training
- Forecasting should be instant
- Reproducibility: same Q-table = same selections

**When to Retrain?**
- New backcast data available (e.g., 2024 actuals)
- Model performance changes
- Different reward function

---

## 8. Performance Analysis

### 8.1 Model Comparison (2019-2023 Backcasts)

| Metric              | SARIMA     | LSTM            | Ridge     | Prophet   |
| ------------------- | ---------- | --------------- | --------- | --------- |
| **Temperature MAE** | 1.106°C    | **1.084°C** ⭐   | 5.174°C   | 4.126°C   |
| **ET0 MAE**         | 17.385 mm  | **15.432 mm** ⭐ | 45.061 mm | 48.739 mm |
| **Combined Score**  | 1.073x     | **1.000x** ⭐    | 3.846x    | 3.482x    |
| **Months Selected** | 68 (33.3%) | 136 (66.7%)     | 0 (0%)    | 0 (0%)    |

**Key Insights:**
- LSTM dominates (best on both variables)
- SARIMA close second (only 7% worse)
- Ridge 3.85x worse (fails on both variables)
- Prophet 3.48x worse (overfits small dataset)

### 8.2 RL Agent Learning Curve

```
Episode    Avg Reward    ε        Notes
-------    ----------    ----     -----
1-100      -60.44        1.0→0.61  High exploration
100-200    -53.84        0.61→0.37 Finding good models
200-500    -53.75        0.37→0.08 Converging
500-1000   -53.67        0.08→0.01 Stable performance
```

**Convergence:** Achieved by episode ~300

**Final Performance:** -53.67 average reward (normalized error)

### 8.3 Forecast Horizon Analysis

**2024-2030 (Near-term):**
- High confidence (6-year extrapolation)
- LSTM/SARIMA reliable

**2030-2040 (Long-term):**
- Moderate confidence (12-16 year extrapolation)
- Trend uncertainty increases
- RL selection still valid (monthly patterns stable)

**Uncertainty Sources:**
1. Model extrapolation error (grows with time)
2. Climate change acceleration (non-linear trends)
3. Extreme events (not captured by monthly averages)

### 8.4 System Performance Metrics

**Training Time:**
- Backcast generation: 10-15 min (Colab GPU)
- RL training: ~1 minute (1000 episodes)
- **Total:** <20 minutes

**Inference Time:**
- RL forecast generation: <1 second
- RAG query: 2-3 seconds
- **User experience:** Real-time

**Storage:**
- Raw data: ~2 MB
- Preprocessed: ~100 KB
- Forecasts: ~500 KB (4 models × 2 variables)
- stats_db.json: ~1.5 MB
- ChromaDB: ~10 MB
- **Total:** <15 MB

---

## 9. Future Enhancements

### 9.1 Uncertainty Quantification

**Current Limitation:** Point forecasts only

**Enhancement:**
```python
# Use Prophet's uncertainty intervals
prophet_forecast = prophet_model.predict(future)
lower_bound = prophet_forecast['yhat_lower']
upper_bound = prophet_forecast['yhat_upper']

# RL agent could prefer models with tighter bands
confidence_penalty = (upper_bound - lower_bound).mean()
reward = -error - 0.1 * confidence_penalty
```

### 9.2 Online Learning

**Current:** Static Q-table

**Enhancement:**
```python
def update_with_actuals(new_actuals_2024):
    # As real 2024 data comes in:
    for month in new_actuals_2024:
        state = month.month - 1
        action = q_table[state, :].argmax()
        
        # Get actual error
        actual = month.temperature
        pred = forecasts[action][month]
        reward = -abs(actual - pred)
        
        # Online Q-update
        q_table[state, action] += lr * (reward - q_table[state, action])
```

### 9.3 Multi-Variable State

**Current:** State = month only

**Enhancement:**
```python
# State = (month, trend_direction, extreme_flag)
state = (
    month,  # 0-11
    1 if temp_trend > 0 else 0,  # Warming/cooling
    1 if abs(temp_anomaly) > 2*std else 0  # Extreme event
)

# Q-table: [12 × 2 × 2, 4] = [48 states, 4 actions]
```

### 9.4 Ensemble Hybrid

**Enhancement:**
```python
# RL selects top 2 models, weighted average
top_2 = q_table[state, :].argsort()[-2:]
weights = softmax(q_table[state, top_2])

final_pred = weights[0] * forecast[top_2[0]] + \
             weights[1] * forecast[top_2[1]]
```

---

## 10. Conclusion

This system demonstrates:
1. **End-to-End ML Pipeline:** Data → Models → Selection → Deployment
2. **Novel RL Application:** Dual-variable Q-learning for climate forecasting
3. **Production-Ready:** RAG interface, persistent storage, error handling
4. **Scientific Rigor:** Real backcasts, proper validation, interpretable decisions

**Key Innovation:** Dual-variable RL training ensures model selection is robust across multiple climate variables simultaneously.

**Production Status:** ✅ Ready for deployment and user queries.