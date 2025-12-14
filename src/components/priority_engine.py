import sys
from src.exception import CustomException
from src.logger import info

class PriorityEngine:
  def __init__(self):

    self.keywords = {
      "Critical" : ['crash','data loss', 'hacked', 'breach', 'emergency', 'shut down', "critical", 'outage'],
      "High" : ['urgent', 'deadline', 'money','refund','charge', 'payment', 'failed', 'immediately'],
      "Medium" : ['error', 'bug', 'glitch', 'slow', 'issue', 'problem', 'delay', 'fix'],
      "Low" : ['question', 'feedback', 'suggestion', "how to", 'inquiry', 'info', 'information', 'help', 'password']
    }


  def predict_priority(self, subject, description):
    try:

      full_text = f"{str(subject)} {str(description)}".lower()  

      for word in self.keywords["Critical"]:
        if word in full_text:
          return "Critical"
        
      for word in self.keywords["High"]:
        if word in full_text:
          return "High"

      for word in self.keywords["Medium"]:
        if word in full_text:
          return "Medium"

      return "Low"

    except Exception as e:
      info("Exception occurred in priority prediction")
      raise CustomException(e, sys)

if __name__ == "__main__":
  engine = PriorityEngine()

  tests = [
      ("System crash", "My server is completely down and I lost data."),
      ("Refund needed", "I was charged twice, please refund immediately."),
      ("Small typo", "There is a spelling mistake on your homepage."),
  ]    

  for sub, desc in tests:
    priority = engine.predict_priority(sub, desc)
    print(f"Subject: {sub}\nDescription: {desc}\nPredicted Priority: {priority}\n")
    
