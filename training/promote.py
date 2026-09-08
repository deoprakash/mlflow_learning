import mlflow
from mlflow import MlflowClient


TRACKING_URI = "http://127.0.0.1:5000"
# TRACKING_URI = "http://host.docker.internal:5000"
MODEL_NAME = "iris-classifier-model"
ALIAS = "champion"

# Minimum requirements for a model to be considered good enough
MIN_ACCURACY = 0.90
MIN_PRECISION = 0.90
MIN_RECALL = 0.90
MIN_F1 = 0.90


mlflow.set_tracking_uri(TRACKING_URI)

print("Tracking URI:", mlflow.get_tracking_uri())

client = MlflowClient()

def get_metrics(version):
    """Get metrics from the run associated with a model version."""

    run = client.get_run(version.run_id)

    return run.data.metrics


def passes_quality_gate(metrics):
    """Check whether the candidate satisfies minimum requirements."""

    return (
        metrics.get("accuracy", 0) >= MIN_ACCURACY
        and metrics.get("precision", 0) >= MIN_PRECISION
        and metrics.get("recall", 0) >= MIN_RECALL
        and metrics.get("f1", 0) >= MIN_F1
    )

def promote_best_model(candidate_version):

    candidate = client.get_model_version(
        MODEL_NAME,
        candidate_version
    )

    candidate_metrics = get_metrics(candidate)

    candidate_accuracy = candidate_metrics.get("accuracy", 0)
    candidate_precision = candidate_metrics.get("precision", 0)
    candidate_recall = candidate_metrics.get("recall", 0)
    candidate_f1 = candidate_metrics.get("f1", 0)

    print()
    print(f"Candidate version: {candidate.version}")

    print("Candidate metrics:")
    print(f"Accuracy:  {candidate_accuracy:.4f}")
    print(f"Precision: {candidate_precision:.4f}")
    print(f"Recall:    {candidate_recall:.4f}")
    print(f"F1:        {candidate_f1:.4f}")

    # --------------------------------------------------
    # STEP 1: QUALITY GATE
    # --------------------------------------------------

    if not passes_quality_gate(candidate_metrics):

        print()
        print("❌ Candidate failed quality gate.")
        print("Champion will remain unchanged.")

        return

    print()
    print("✅ Candidate passed quality gate.")

    # --------------------------------------------------
    # STEP 2: GET CURRENT CHAMPION
    # --------------------------------------------------

    try:

        champion = client.get_model_version_by_alias(
            MODEL_NAME,
            ALIAS
        )

        champion_metrics = get_metrics(champion)

        champion_f1 = champion_metrics.get("f1", 0)

        print()
        print(f"Current champion version: {champion.version}")
        print(f"Champion F1: {champion_f1:.4f}")

    except Exception:

        champion = None
        champion_f1 = None

        print()
        print("No champion currently exists.")

    # --------------------------------------------------
    # STEP 3: NO CHAMPION -> PROMOTE
    # --------------------------------------------------

    if champion is None:

        client.set_registered_model_alias(
            MODEL_NAME,
            ALIAS,
            candidate.version
        )

        print()
        print(
            f"✅ Version {candidate.version} "
            f"promoted to champion."
        )

        return

    # --------------------------------------------------
    # STEP 4: COMPARE CANDIDATE VS CHAMPION
    # --------------------------------------------------

    if candidate_f1 >= champion_f1:

        client.set_registered_model_alias(
            MODEL_NAME,
            ALIAS,
            candidate.version
        )

        print()
        print(
            f"✅ Version {candidate.version} "
            f"promoted to champion."
        )

        print(
            f"Candidate F1: {candidate_f1:.4f}"
        )

        print(
            f"Previous champion F1: {champion_f1:.4f}"
        )

    else:

        print()
        print(
            f"❌ Version {candidate.version} "
            f"was NOT promoted."
        )

        print(
            f"Candidate F1: {candidate_f1:.4f}"
        )

        print(
            f"Champion F1: {champion_f1:.4f}"
        )

        print(
            f"Champion remains version {champion.version}."
        )


if __name__ == "__main__":
    print("promote.py should be called from pipeline.py")