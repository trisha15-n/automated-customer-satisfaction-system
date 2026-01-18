# Intelligent Customer Support System

An automated triage system for Customer Support teams. This application uses a **Hybrid AI Architecture** (Machine Learning + Rule-Based Logic) to classify tickets, detect urgency, and predict customer churn risk in real-time.

##  Key Features

* ** Automated Classification:** Uses ML to categorize tickets (e.g., Technical Support, Billing, Hardware Issues).
* **Smart Priority Detection:** Combines **Machine Learning** with a **Rule-Based Engine** to catch critical issues (e.g., "System Crash", "Hacked") instantly, ensuring 100% compliance on safety-critical tickets.
* **Churn Risk Assessment:** Analyzes customer sentiment (VADER) and ticket metadata to predict the likelihood of the customer leaving.
* **Guardrail System:** Intelligent logic to detect "Positive Feedback" or "Routine Requests" (like "Send Invoice") and bypass unnecessary alarms.
* **Clean Architecture:** Built using a **Pipeline Pattern** (Facade Design Pattern) to completely separate Business Logic from the UI.

## Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python 3.8+
* **Machine Learning:** Scikit-Learn, Pandas, Numpy
* **NLP:** NLTK (VADER Sentiment Analysis)
* **Logging:** Custom Logger & Exception Handling

