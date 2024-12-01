from main import logger
from sklearn.metrics import classification_report,f1_score


def evaluate_model(preds,y_true):
        
    maf = f1_score(y_true, preds, average='macro',zero_division=0)
    waf = f1_score(y_true, preds, average='weighted',zero_division=0)
        
    report = classification_report(y_true, preds, zero_division=0)
    logger.info("Evaluation Report:\n", report)
    logger.info(f"Macro F1 Score: {maf:.4f}")
    logger.info(f"Weighted F1 Score: {waf:.4f}")
        
    # with open('/root/DOMR_torch/experiment/logs/evaluation_report.log','w') as f:
    #     f.write("Evaluation Report:\n")
    #     f.write(report)
    #     f.write(f"Macro F1 Score: {maf:.4f}\n")
    #     f.write(f"Weighted F1 Score: {waf:.4f}\n")
