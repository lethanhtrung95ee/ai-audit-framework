import json
import os

def save_report(report_data: dict, output_path="reports/report.json"):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    from datetime import datetime
    report_data["generated_at"] = str(datetime.utcnow())
    with open(output_path, 'w') as f:
        json.dump(report_data, f, indent=4)

