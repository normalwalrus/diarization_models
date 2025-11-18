import os
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate


def compute_der_for_folder(pred_dir, gold_dir):
    # Explicit scoring options (matches AMI-style scoring)
    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)

    pred_files = sorted(f for f in os.listdir(pred_dir) if f.endswith(".rttm"))

    for fname in pred_files:
        hyp_path = os.path.join(pred_dir, fname)
        ref_path = os.path.join(gold_dir, fname)

        if not os.path.exists(ref_path):
            print(f"[WARN] Missing reference RTTM for {fname}, skipping...")
            continue

        # Each load_rttm returns a dict {uri: Annotation}
        hyp_dict = load_rttm(hyp_path)
        ref_dict = load_rttm(ref_path)

        # RTTM files should contain exactly one URI; extract it
        hyp_uri = next(iter(hyp_dict))
        ref_uri = next(iter(ref_dict))

        hypothesis = hyp_dict[hyp_uri]
        reference = ref_dict[ref_uri]

        metric(reference, hypothesis)

    report = metric.report()
    global_der = report.loc["TOTAL", "diarization error rate"].iloc[0]
    return global_der, report


if __name__ == "__main__":
    pred_dir = "outputs/pyannote_community_1/NSC_PART3"
    gold_dir = "/home/digitalhub/Desktop/data/diarization_benchmarks/NSC_PART3_SPON_DIARISATION/rttm"

    der, report = compute_der_for_folder(pred_dir, gold_dir)

    print("===== Global DER over all files =====")
    print(f"DER: {der:.5f}")
    print(report)
