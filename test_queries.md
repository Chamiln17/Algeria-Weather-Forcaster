# RAG System Test Queries

## 1. Specific Date/Month Queries

### Query 1.1: Specific Month in Specific Year
**Question:** "What's the temperature forecast for February 2026?"  
**Expected:** Direct answer: ~11.87°C (SARIMA)

### Query 1.2: Different Phrasing
**Question:** "Tell me about February 2026 temperature"  
**Expected:** Same as 1.1

### Query 1.3: Far Future Date
**Question:** "What will the temperature be in December 2040?"  
**Expected:** ~15.73°C

### Query 1.4: Near-Term Date
**Question:** "Temperature forecast for March 2024?"  
**Expected:** ~13.66°C

### Query 1.5: Summer Month
**Question:** "What about July 2030?"  
**Expected:** ~25.7°C (high summer temp)

### Query 1.6: Winter Month
**Question:** "January 2035 temperature?"  
**Expected:** ~12.5-12.7°C (low winter temp)

---

## 2. Year-Range Queries

### Query 2.1: Single Year Overview
**Question:** "Give me the temperature forecast for 2026"  
**Expected:** 12 monthly values from Jan-Dec 2026

### Query 2.2: Multi-Year Trend
**Question:** "What are the temperature trends from 2024 to 2040?"  
**Expected:** Should mention beginning (2024), middle (~2032), and end (2040) data points

### Query 2.3: Short Range
**Question:** "Temperature changes between 2025 and 2027"  
**Expected:** Data from 2025, 2026, 2027

### Query 2.4: Specific Period
**Question:** "Show me forecasts for 2035-2038"  
**Expected:** Focus on those specific years

---

## 3. ET0 Queries (Testing Label Differentiation)

### Query 3.1: Specific ET0 Query
**Question:** "What is the ET0 forecast for February 2026?"  
**Expected:** ~77 mm (should NOT confuse with temperature 11.87°C)

### Query 3.2: ET0 Trend
**Question:** "How does ET0 change from 2024 to 2040?"  
**Expected:** Range from ~57 mm to ~197 mm

### Query 3.3: Summer ET0
**Question:** "ET0 in July 2030?"  
**Expected:** High value ~188 mm (summer peak)

### Query 3.4: Winter ET0
**Question:** "What about ET0 in January 2025?"  
**Expected:** Low value ~57 mm

---

## 4. Combined Temperature + ET0 Queries

### Query 4.1: Both Variables
**Question:** "Talk about temperature and ET0 trends from 2024-2040"  
**Expected:** 
- Temperature: 11-27°C range
- ET0: 57-197 mm range
- Should NOT mix values (no "183°C")

### Query 4.2: Specific Month, Both Variables
**Question:** "What are temperature and ET0 forecasts for June 2027?"  
**Expected:**
- Temperature: ~23°C
- ET0: ~160 mm
- Clear separation

### Query 4.3: Comparison Query
**Question:** "Compare summer vs winter for temperature and ET0"  
**Expected:**
- Summer: High temp (~26-27°C), High ET0 (~190-197 mm)
- Winter: Low temp (~11-12°C), Low ET0 (~57-60 mm)

---

## 5. Summary Statistics Queries

### Query 5.1: Overall Stats
**Question:** "What are the overall temperature statistics for Algeria?"  
**Expected:** Mean 18.7°C, range 11.1-27.3°C

### Query 5.2: ET0 Stats
**Question:** "What are the ET0 statistics?"  
**Expected:** Mean ~121 mm, range 56.7-197 mm

### Query 5.3: Variability
**Question:** "How variable are the temperature forecasts?"  
**Expected:** Std dev ~5.05°C

---

## 6. Edge Cases & Data Availability

### Query 6.1: First Date in Dataset
**Question:** "What's the temperature forecast for January 2024?"  
**Expected:** ~11.12°C (min value)

### Query 6.2: Last Date in Dataset
**Question:** "Temperature in December 2040?"  
**Expected:** ~15.73°C

### Query 6.3: Out-of-Range Query
**Question:** "What about 2041?"  
**Expected:** "Data not available - forecasts only cover 2024-2040"

### Query 6.4: Out-of-Range Query (Past)
**Question:** "Temperature in 2023?"  
**Expected:** "Data not available"

### Query 6.5: Monthly Breakdown Not Available
**Question:** "What's the temperature on January 15, 2026?"  
**Expected:** "Only monthly data available, not daily"

---

## 7. Seasonal & Comparative Queries

### Query 7.1: Hottest Period
**Question:** "When is the hottest period in Algeria?"  
**Expected:** July-August, ~26-27°C

### Query 7.2: Coldest Period
**Question:** "When is it coldest?"  
**Expected:** January, ~11-12°C

### Query 7.3: Year-over-Year
**Question:** "Is 2040 warmer than 2024?"  
**Expected:** Similar (stable climate), both ~same range

### Query 7.4: Seasonal ET0
**Question:** "How does ET0 vary by season?"  
**Expected:** Low in winter (~57), high in summer (~197)

---

## 8. Phrasing Variations (Robustness Testing)

### Query 8.1: Casual Phrasing
**Question:** "how hot will it be in august 2030?"  
**Expected:** ~26.8°C

### Query 8.2: Formal Phrasing
**Question:** "Could you provide the mean temperature forecast for the month of August in the year 2030?"  
**Expected:** Same as 8.1

### Query 8.3: Abbreviated
**Question:** "temp feb 26"  
**Expected:** Should understand as February 2026

### Query 8.4: Verbose
**Question:** "I would like to know what the forecasted temperature will be for Algeria in the month of February during the year 2026"  
**Expected:** Still ~11.87°C

---

## 9. Model-Related Queries

### Query 9.1: Which Model Used
**Question:** "What model predicts February 2026 temperature?"  
**Expected:** SARIMA

### Query 9.2: Model Distribution
**Question:** "Which models are used most often?"  
**Expected:** LSTM (136 times), SARIMA (68 times)

### Query 9.3: Multiple Models
**Question:** "Do different models predict different values?"  
**Expected:** Yes, RL agent selects the best one

---

## 10. Potential Confusion Tests (Critical!)

### Query 10.1: Temperature Only (No ET0 Mixing)
**Question:** "What are the temperature trends?"  
**Expected:** 
- ✅ Values: 11-27°C
- ❌ Should NOT mention: 57°C, 183°C, 197°C

### Query 10.2: ET0 Only (No Temperature Mixing)
**Question:** "What are the ET0 trends?"  
**Expected:**
- ✅ Values: 57-197 mm
- ❌ Should NOT say: "11°C ET0" or "27 mm temperature"

### Query 10.3: Explicit Separation
**Question:** "Give me both temperature and ET0 for 2026"  
**Expected:**
- Temperature: 11-27°C range
- ET0: 56-196 mm range
- NO cross-contamination

---

## Success Criteria

| Test Category          | Pass Criteria                                  |
| ---------------------- | ---------------------------------------------- |
| **Specific Queries**   | Returns exact value ±0.5°C/mm                  |
| **Range Queries**      | Includes beginning, middle, end data points    |
| **ET0 vs Temperature** | NO mixing of values (e.g., no "183°C")         |
| **Units**              | Always includes °C for temp, mm for ET0        |
| **Out-of-Range**       | Clearly states data not available              |
| **Seasonal**           | Correctly identifies summer highs, winter lows |
| **Robustness**         | Handles different phrasings consistently       |

---

## How to Test

1. **Manual Testing:** Paste queries one-by-one into Streamlit chat
2. **Automated Testing:** Use the script below

```python
# test_rag.py
from src.rag import init_rag_system
import os

queries = [
    "What's the temperature forecast for February 2026?",
    "What is the ET0 forecast for February 2026?",
    "Talk about temperature and ET0 trends from 2024-2040",
    # ... add all queries
]

rag = init_rag_system(os.environ['GROQ_API_KEY'])

for i, query in enumerate(queries, 1):
    print(f"\n{'='*60}")
    print(f"Query {i}: {query}")
    print(f"{'='*60}")
    response = rag.query(query)
    print(response)
```

---

## Expected Document Count After Re-embedding

With the new labeling:
- **2 Summary Documents** (RL_Agent_Temperature, RL_Agent_ET0)
- **17 Temperature Yearly Documents** (2024-2040)
- **17 ET0 Yearly Documents** (2024-2040)
- **Total: 36 documents**
