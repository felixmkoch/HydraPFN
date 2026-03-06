
from hydrapfn.scripts.tabarena_evaluator import TabArenaEvaluator
from hydrapfn.scripts.model_loader import load_hydrapfn_model

MODEL_PATH = "hydrapfn/trained_models/test_model.cpkt"

# Load model
print(f"Loading model from: {MODEL_PATH}")
model, optimizer, config = load_hydrapfn_model(MODEL_PATH, device="cuda")

evaluator = TabArenaEvaluator()

evaluator.eval_on_tabarena(model)