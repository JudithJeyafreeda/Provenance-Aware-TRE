{
  "config": {
    "model_name": "emilyalsentzer/Bio_ClinicalBERT",
    "seed": 7,
    "out_dir": "output",
    "epochs": 20,
    "batch_size": 8,
    "lr": 5e-05,
    "max_len": 128,
    "finetune_bert": false,
    "min_coverage": 0.08,
    "min_coverage_target": 0.1,
    "min_coverage_stop_threshold": 0.08,
    "lambda_evi": 0.0005,
    "lambda_coverage_floor": 1.0,
    "lambda_faith": 0.5,
    "lambda_causal": 0.1,
    "warmup_epochs": 6,
    "tau": 0.5,
    "run_certification": true,
    "cert_n_examples": 30,
    "cert_n_samples": 100,
    "cert_alpha": 0.01,
    "cert_epsilon_budget": 2,
    "cert_theta_stability": 0.5,
    "site_id": "bert_site",
    "data_dir": "/mnt/eds_projets/eds_iam/Judith_ARIPPA/work/LLM/TEMPORALEX/output/synthetic_dataset"
  },
  "test_metrics": {
    "baseline": {
      "accuracy_on_answered": 0.74,
      "abstain_rate": 0.0,
      "mean_evidence_recall": 0.0,
      "mean_coverage": 0.03187447263548771,
      "n": 150,
      "per_template_breakdown": {
        "P00010": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00024": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00041": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00044": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00051": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00057": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00069": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00078": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00081": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00085": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00089": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00090": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00098": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00114": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00116": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00131": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00139": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00143": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00151": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00160": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00163": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00178": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00179": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00188": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00193": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00194": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00196": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00197": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00202": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00203": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00206": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00208": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00211": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00214": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00215": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00218": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00222": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00230": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00240": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00241": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00247": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00255": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00261": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00264": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00273": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00277": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00289": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00295": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00308": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00313": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00314": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00327": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00334": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00338": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00342": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00349": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00359": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00371": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00382": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00386": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00412": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00414": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00415": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00416": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00422": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00423": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00442": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00443": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00445": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00449": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00460": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00461": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00465": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00468": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00475": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00487": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00488": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00492": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00501": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00510": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00531": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00533": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00534": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00542": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00565": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00581": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00594": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00609": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00612": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00613": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00617": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00640": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00652": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00653": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00663": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00683": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00685": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00687": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00688": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00695": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00721": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00735": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00743": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00747": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00754": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00760": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00761": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00774": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00783": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00785": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00789": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00792": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00793": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00804": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00811": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00812": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00822": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00825": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00831": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00834": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00838": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00843": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00845": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00847": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00850": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00852": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00857": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00864": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00866": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00870": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00874": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00878": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00885": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00886": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00910": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00911": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00927": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00933": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00943": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00946": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00947": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00956": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00958": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00962": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00967": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00971": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00991": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00994": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00996": {
          "n": 1,
          "abstained": 0,
          "correct": 0
        },
        "P00999": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        }
      },
      "per_template_abstain_rate": {
        "P00010": 0.0,
        "P00024": 0.0,
        "P00041": 0.0,
        "P00044": 0.0,
        "P00051": 0.0,
        "P00057": 0.0,
        "P00069": 0.0,
        "P00078": 0.0,
        "P00081": 0.0,
        "P00085": 0.0,
        "P00089": 0.0,
        "P00090": 0.0,
        "P00098": 0.0,
        "P00114": 0.0,
        "P00116": 0.0,
        "P00131": 0.0,
        "P00139": 0.0,
        "P00143": 0.0,
        "P00151": 0.0,
        "P00160": 0.0,
        "P00163": 0.0,
        "P00178": 0.0,
        "P00179": 0.0,
        "P00188": 0.0,
        "P00193": 0.0,
        "P00194": 0.0,
        "P00196": 0.0,
        "P00197": 0.0,
        "P00202": 0.0,
        "P00203": 0.0,
        "P00206": 0.0,
        "P00208": 0.0,
        "P00211": 0.0,
        "P00214": 0.0,
        "P00215": 0.0,
        "P00218": 0.0,
        "P00222": 0.0,
        "P00230": 0.0,
        "P00240": 0.0,
        "P00241": 0.0,
        "P00247": 0.0,
        "P00255": 0.0,
        "P00261": 0.0,
        "P00264": 0.0,
        "P00273": 0.0,
        "P00277": 0.0,
        "P00289": 0.0,
        "P00295": 0.0,
        "P00308": 0.0,
        "P00313": 0.0,
        "P00314": 0.0,
        "P00327": 0.0,
        "P00334": 0.0,
        "P00338": 0.0,
        "P00342": 0.0,
        "P00349": 0.0,
        "P00359": 0.0,
        "P00371": 0.0,
        "P00382": 0.0,
        "P00386": 0.0,
        "P00412": 0.0,
        "P00414": 0.0,
        "P00415": 0.0,
        "P00416": 0.0,
        "P00422": 0.0,
        "P00423": 0.0,
        "P00442": 0.0,
        "P00443": 0.0,
        "P00445": 0.0,
        "P00449": 0.0,
        "P00460": 0.0,
        "P00461": 0.0,
        "P00465": 0.0,
        "P00468": 0.0,
        "P00475": 0.0,
        "P00487": 0.0,
        "P00488": 0.0,
        "P00492": 0.0,
        "P00501": 0.0,
        "P00510": 0.0,
        "P00531": 0.0,
        "P00533": 0.0,
        "P00534": 0.0,
        "P00542": 0.0,
        "P00565": 0.0,
        "P00581": 0.0,
        "P00594": 0.0,
        "P00609": 0.0,
        "P00612": 0.0,
        "P00613": 0.0,
        "P00617": 0.0,
        "P00640": 0.0,
        "P00652": 0.0,
        "P00653": 0.0,
        "P00663": 0.0,
        "P00683": 0.0,
        "P00685": 0.0,
        "P00687": 0.0,
        "P00688": 0.0,
        "P00695": 0.0,
        "P00721": 0.0,
        "P00735": 0.0,
        "P00743": 0.0,
        "P00747": 0.0,
        "P00754": 0.0,
        "P00760": 0.0,
        "P00761": 0.0,
        "P00774": 0.0,
        "P00783": 0.0,
        "P00785": 0.0,
        "P00789": 0.0,
        "P00792": 0.0,
        "P00793": 0.0,
        "P00804": 0.0,
        "P00811": 0.0,
        "P00812": 0.0,
        "P00822": 0.0,
        "P00825": 0.0,
        "P00831": 0.0,
        "P00834": 0.0,
        "P00838": 0.0,
        "P00843": 0.0,
        "P00845": 0.0,
        "P00847": 0.0,
        "P00850": 0.0,
        "P00852": 0.0,
        "P00857": 0.0,
        "P00864": 0.0,
        "P00866": 0.0,
        "P00870": 0.0,
        "P00874": 0.0,
        "P00878": 0.0,
        "P00885": 0.0,
        "P00886": 0.0,
        "P00910": 0.0,
        "P00911": 0.0,
        "P00927": 0.0,
        "P00933": 0.0,
        "P00943": 0.0,
        "P00946": 0.0,
        "P00947": 0.0,
        "P00956": 0.0,
        "P00958": 0.0,
        "P00962": 0.0,
        "P00967": 0.0,
        "P00971": 0.0,
        "P00991": 0.0,
        "P00994": 0.0,
        "P00996": 0.0,
        "P00999": 0.0
      },
      "shortcut_flag": false
    },
    "provenance_first": {
      "accuracy_on_answered": 1.0,
      "abstain_rate": 0.96,
      "mean_evidence_recall": 0.0,
      "mean_coverage": 0.025419932194054126,
      "n": 150,
      "per_template_breakdown": {
        "P00010": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00024": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00041": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00044": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00051": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00057": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00069": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00078": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00081": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00085": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00089": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00090": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00098": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00114": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00116": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00131": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00139": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00143": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00151": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00160": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00163": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00178": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00179": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00188": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00193": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00194": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00196": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00197": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00202": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00203": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00206": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00208": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00211": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00214": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00215": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00218": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00222": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00230": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00240": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00241": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00247": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00255": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00261": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00264": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00273": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00277": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00289": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00295": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00308": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00313": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00314": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00327": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00334": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00338": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00342": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00349": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00359": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00371": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00382": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00386": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00412": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00414": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00415": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00416": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00422": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00423": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00442": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00443": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00445": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00449": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00460": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00461": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00465": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00468": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00475": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00487": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00488": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00492": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00501": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00510": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00531": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00533": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00534": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00542": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00565": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00581": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00594": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00609": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00612": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00613": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00617": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00640": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00652": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00653": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00663": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00683": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00685": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00687": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00688": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00695": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00721": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00735": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00743": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00747": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00754": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00760": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00761": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00774": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00783": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00785": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00789": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00792": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00793": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00804": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00811": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00812": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00822": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00825": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00831": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00834": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00838": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00843": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00845": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00847": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00850": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00852": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00857": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00864": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00866": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00870": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00874": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00878": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00885": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00886": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00910": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00911": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00927": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00933": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00943": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00946": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00947": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00956": {
          "n": 1,
          "abstained": 0,
          "correct": 1
        },
        "P00958": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00962": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00967": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00971": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00991": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00994": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00996": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        },
        "P00999": {
          "n": 1,
          "abstained": 1,
          "correct": 0
        }
      },
      "per_template_abstain_rate": {
        "P00010": 1.0,
        "P00024": 1.0,
        "P00041": 1.0,
        "P00044": 1.0,
        "P00051": 1.0,
        "P00057": 1.0,
        "P00069": 1.0,
        "P00078": 0.0,
        "P00081": 1.0,
        "P00085": 1.0,
        "P00089": 1.0,
        "P00090": 1.0,
        "P00098": 1.0,
        "P00114": 1.0,
        "P00116": 1.0,
        "P00131": 1.0,
        "P00139": 1.0,
        "P00143": 1.0,
        "P00151": 1.0,
        "P00160": 1.0,
        "P00163": 1.0,
        "P00178": 1.0,
        "P00179": 1.0,
        "P00188": 1.0,
        "P00193": 1.0,
        "P00194": 1.0,
        "P00196": 1.0,
        "P00197": 1.0,
        "P00202": 1.0,
        "P00203": 1.0,
        "P00206": 1.0,
        "P00208": 1.0,
        "P00211": 1.0,
        "P00214": 1.0,
        "P00215": 1.0,
        "P00218": 1.0,
        "P00222": 1.0,
        "P00230": 1.0,
        "P00240": 1.0,
        "P00241": 1.0,
        "P00247": 1.0,
        "P00255": 1.0,
        "P00261": 1.0,
        "P00264": 1.0,
        "P00273": 1.0,
        "P00277": 1.0,
        "P00289": 1.0,
        "P00295": 1.0,
        "P00308": 1.0,
        "P00313": 1.0,
        "P00314": 1.0,
        "P00327": 1.0,
        "P00334": 1.0,
        "P00338": 1.0,
        "P00342": 1.0,
        "P00349": 1.0,
        "P00359": 1.0,
        "P00371": 1.0,
        "P00382": 1.0,
        "P00386": 1.0,
        "P00412": 1.0,
        "P00414": 1.0,
        "P00415": 1.0,
        "P00416": 0.0,
        "P00422": 1.0,
        "P00423": 1.0,
        "P00442": 1.0,
        "P00443": 1.0,
        "P00445": 1.0,
        "P00449": 1.0,
        "P00460": 1.0,
        "P00461": 1.0,
        "P00465": 1.0,
        "P00468": 1.0,
        "P00475": 1.0,
        "P00487": 1.0,
        "P00488": 1.0,
        "P00492": 1.0,
        "P00501": 1.0,
        "P00510": 1.0,
        "P00531": 1.0,
        "P00533": 0.0,
        "P00534": 1.0,
        "P00542": 1.0,
        "P00565": 1.0,
        "P00581": 1.0,
        "P00594": 1.0,
        "P00609": 1.0,
        "P00612": 1.0,
        "P00613": 1.0,
        "P00617": 1.0,
        "P00640": 1.0,
        "P00652": 0.0,
        "P00653": 1.0,
        "P00663": 1.0,
        "P00683": 1.0,
        "P00685": 1.0,
        "P00687": 1.0,
        "P00688": 1.0,
        "P00695": 1.0,
        "P00721": 1.0,
        "P00735": 1.0,
        "P00743": 1.0,
        "P00747": 1.0,
        "P00754": 1.0,
        "P00760": 1.0,
        "P00761": 1.0,
        "P00774": 1.0,
        "P00783": 1.0,
        "P00785": 1.0,
        "P00789": 1.0,
        "P00792": 1.0,
        "P00793": 1.0,
        "P00804": 1.0,
        "P00811": 1.0,
        "P00812": 1.0,
        "P00822": 1.0,
        "P00825": 1.0,
        "P00831": 1.0,
        "P00834": 1.0,
        "P00838": 1.0,
        "P00843": 1.0,
        "P00845": 1.0,
        "P00847": 1.0,
        "P00850": 1.0,
        "P00852": 1.0,
        "P00857": 1.0,
        "P00864": 1.0,
        "P00866": 1.0,
        "P00870": 1.0,
        "P00874": 1.0,
        "P00878": 1.0,
        "P00885": 1.0,
        "P00886": 1.0,
        "P00910": 1.0,
        "P00911": 0.0,
        "P00927": 1.0,
        "P00933": 1.0,
        "P00943": 1.0,
        "P00946": 1.0,
        "P00947": 1.0,
        "P00956": 0.0,
        "P00958": 1.0,
        "P00962": 1.0,
        "P00967": 1.0,
        "P00971": 1.0,
        "P00991": 1.0,
        "P00994": 1.0,
        "P00996": 1.0,
        "P00999": 1.0
      },
      "shortcut_flag": true
    }
  },
  "position_control_metrics": null,
  "manipulation_metrics": {
    "baseline": {
      "adversarial_n": 1887,
      "adversarial_abstain_rate": 0.0,
      "adversarial_pred_stability": 0.9067302596714362,
      "adversarial_evidence_shift_rate": 0.041865394806571275,
      "natural_n": 0,
      "natural_abstain_rate": 0.0,
      "natural_pred_stability": 0.0,
      "natural_mean_evidence_recall": null
    },
    "provenance_first": {
      "adversarial_n": 1887,
      "adversarial_abstain_rate": 0.9899311075781664,
      "adversarial_pred_stability": 0.00688924218335983,
      "adversarial_evidence_shift_rate": 0.00794912559618442,
      "natural_n": 0,
      "natural_abstain_rate": 0.0,
      "natural_pred_stability": 0.0,
      "natural_mean_evidence_recall": null
    }
  },
  "certification": {
    "baseline": {
      "n_examples": 30,
      "n_certified": 0,
      "n_base_abstained": 0,
      "certification_rate": 0.0,
      "mean_prediction_confidence": 0.9256567942861287,
      "mean_provenance_confidence": 0.5673361742413181,
      "per_example": [
        {
          "status": "UNCERTIFIED",
          "prediction_confidence": 0.8793673007031011,
          "provenance_confidence": 0.6562714059342818,
          "p_pred_lower": 0.8793673007031011,
          "p_evi_lower": 0.6562714059342818,
          "n_pred_success": 96,
          "n_evi_success": 78,
          "n_samples": 100,
          "alpha": 0.01,
          "theta_stability": 0.5,
          "epsilon_budget": 2,
          "perturbation_space": [
            "abbreviation_swap",
            "ocr_noise",
            "token_insertion",
            "synonym_substitution"
          ],
          "perturbation_counts": {
            "abbreviation_swap": 26,
            "token_insertion": 25,
            "synonym_substitution": 28,
            "ocr_noise": 21
          },
          "reason": "prediction",
          "note_id": "P00811_N00"
        },
        {
          "status": "UNCERTIFIED",
          "prediction_confidence": 0.948395970375896,
          "provenance_confidence": 0.5172209835264601,
          "p_pred_lower": 0.948395970375896,
          "p_evi_lower": 0.5172209835264601,
          "n_pred_success": 100,
          "n_evi_success": 65,
          "n_samples": 100,
          "alpha": 0.01,
          "theta_stability": 0.5,
          "epsilon_budget": 2,
          "perturbation_space": [
            "abbreviation_swap",
            "ocr_noise",
            "token_insertion",
            "synonym_substitution"
          ],
          "perturbation_counts": {
            "synonym_substitution": 26,
            "ocr_noise": 23,
            "abbreviation_swap": 24,
            "token_insertion": 27
          },
          "reason": "prediction",
          "note_id": "P00415_N00"
        },
        {
          "status": "UNCERTIFIED",
          "prediction_confidence": 0.8793673007031011,
          "provenance_confidence": 0.4564833738464764,
          "p_pred_lower": 0.8793673007031011,
          "p_evi_lower": 0.4564833738464764,
          "n_pred_success": 96,
          "n_evi_success": 59,
          "n_samples": 100,
          "alpha": 0.01,
          "theta_stability": 0.5,
          "epsilon_budget": 2,
          "perturbation_space": [
            "abbreviation_swap",
            "ocr_noise",
            "token_insertion",
            "synonym_substitution"
          ],
          "perturbation_counts": {
            "token_insertion": 30,
            "synonym_substitution": 27,
            "ocr_noise": 21,
            "abbreviation_swap": 22
          },
          "reason": "prediction",
          "note_id": "P00214_N00"
        },
        {
          "status": "UNCERTIFIED",
          "prediction_confidence": 0.948395970375896,
          "provenance_confidence": 0.44654555318206685,
          "p_pred_lower": 0.948395970375896,
          "p_evi_lower": 0.44654555318206685,
          "n_pred_success": 100,
          "n_evi_success": 58,
          "n_samples": 100,
          "alpha": 0.01,
          "theta_stability": 0.5,
          "epsilon_budget": 2,
          "perturbation_space": [
            "abbreviation_swap",
            "ocr_noise",
            "token_insertion",
            "synonym_substitution"
          ],
          "perturbation_counts": {
            "synonym_substitution": 23,
            "ocr_noise": 28,
            "token_insertion": 21,
            "abbreviation_swap": 28
          },
          "reason": "prediction",
          "note_id": "P00774_N00"
        },
        {
          "status": "UNCERTIFIED",
          "prediction_confidence": 0.9280423175983636,
          "provenance_confidence": 0.6451489444766687,
          "p_pred_lower": 0.9280423175983636,
          "p_evi_lower": 0.6451489444766687,
          "n_pred_success": 99,
          "n_evi_success": 77,
          "n_samples": 100,
          "alpha": 0.01,
          "theta_stability": 0.5,
          "epsilon_budget": 2,
          "perturbation_space": [
            "abbreviation_swap",
            "ocr_noise",
            "token_insertion",
            "synonym_substitution"
          ],
          "perturbation_counts": {
            "token_insertion": 23,
            "abbreviation_swap": 26,
            "synonym_substitution": 18,
            "ocr_noise": 33
          },
          "reason": "prediction",
          "note_id": "P00078_N00"
        },
        {
          "status": "UNCERTIFIED",
          "prediction_confidence": 0.948395970375896,
          "provenance_confidence": 0.5800033176932302,
          "p_pred_lower": 0.948395970375896,
          "p_evi_lower": 0.5800033176932302,
          "n_pred_success": 100,
          "n_evi_success": 71,
          "n_samples": 100,
          "alpha": 0.01,
          "theta_stability": 0.5,
          "epsilon_budget": 2,
          "perturbation_space": [
            "abbreviation_swap",
            "ocr_noise",
            "token_insertion",
            "synonym_substitution"
          ],
          "perturbation_counts": {
            "token_insertion": 25,
            "synonym_substitution": 20,
            "ocr_noise": 22,
            "abbreviation_swap": 33
          },
          "reason": "prediction",
          "note_id": "P00721_N00"
        },
        {
          "status": "UNCERTIFIED",
          "prediction_confidence": 0.948395970375896,
          "provenance_confidence": 0.9280423175983636,
          "p_pred_lower": 0.948395970375896,
          "p_evi_lower": 0.9280423175983636,
          "n_pred_success": 100,
          "n_evi_success": 99,
          "n_samples": 100,
          "alpha": 0.01,
          "theta_stability": 0.5,
          "epsilon_budget": 2,
          "perturbation_space": [
            "abbreviation_swap",
            "ocr_noise",
            "token_insertion",
            "synonym_substitution"
          ],
          "perturbation_counts": {
            "token_insertion": 23,
            "abbreviation_swap": 26,
            "ocr_noise": 22,
            "synonym_substitution": 29
          },
          "reason": "prediction",
          "note_id": "P00247_N00"
        },
        {
          "status": "UNCERTIFIED",
          "prediction_confidence": 0.948395970375896,
          "provenance_confidence": 0.9105693270134086,
          "p_pred_lower": 0.948395970375896,
          "p_evi_lower": 0.9105693270134086,
          "n_pred_success": 100,
          "n_evi_success": 98,
          "n_samples": 100,
          "alpha": 0.01,
          "theta_stability": 0.5,
          "epsilon_budget": 2,
          "perturbation_space": [
            "abbreviation_swap",
            "ocr_noise",
            "token_insertion",
            "synonym_substitution"
          ],
          "perturbation_counts": {
            "token_insertion": 24,
            "abbreviation_swap": 27,
            "synonym_substitution": 23,
            "ocr_noise": 26
          },
          "reason": "prediction",
          "note_id": "P00594_N00"
        },
        {
          "status": "UNCERTIFIED",
          "prediction_confidence": 0.948395970375896,
          "provenance_confidence": 0.9105693270134086,
          "p_pred_lower": 0.948395970375896,
          "p_evi_lower": 0.9105693270134086,
          "n_pred_success": 100,
          "n_evi_success": 98,
          "n_samples": 100,
          "alpha": 0.01,
          "theta_stability": 0.5,
          "epsilon_budget": 2,
          "perturbation_space": [
            "abbreviation_swap",
            "ocr_noise",
            "token_insertion",
            "synonym_substitution"
          ],
          "perturbation_counts": {
            "token_insertion": 24,
            "synonym_substitution": 18,
            "abbreviation_swap": 28,
            "ocr_noise": 30
          },
          "reason": "prediction",
          "note_id": "P00488_N00"
        },
        {
          "status": "UNCERTIFIED",
          "prediction_confidence": 0.948395970375896,
          "provenance_confidence": 0.42682337513434726,
          "p_pred_lower": 0.948395970375896,
          "p_evi_lower": 0.42682337513434726,
          "n_pred_success": 100,
          "n_evi_success": 56,
          "n_samples": 100,
          "alpha": 0.01,
          "theta_stability": 0.5,
          "epsilon_budget": 2,
          "perturbation_space": [
            "abbreviation_swap",
            "ocr_noise",
            "token_insertion",
            "synonym_substitution"
          ],
          "perturbation_counts": {
            "synonym_substitution": 33,
            "token_insertion": 31,
            "ocr_noise": 19,
            "abbreviation_swap": 17
          },
          "reason": "prediction",
          "note_id": "P00461_N00"
        },
        {
          "status": "UNCERTIFIED",
          "prediction_confidence": 0.948395970375896,
          "provenance_confidence": 0.417038217305534,
          "p_pred_lower": 0.948395970375896,
          "p_evi_lower": 0.417038217305534,
          "n_pred_success": 100,
          "n_evi_success": 55,
          "n_samples": 100,
          "alpha": 0.01,
          "theta_stability": 0.5,
          "epsilon_budget": 2,
          "perturbation_space": [
            "abbreviation_swap",
            "ocr_noise",
            "token_insertion",
            "synonym_substitution"
          ],
          "perturbation_counts": {
            "ocr_noise": 28,
            "token_insertion": 22,
            "abbreviation_swap": 29,
            "synonym_substitution": 21
          },
          "reason": "prediction",
          "note_id": "P00910_N00"
        },
        {
          "status": "UNCERTIFIED",
          "prediction_confidence": 0.948395970375896,
          "provenance_confidence": 0.24948706677473803,
          "p_pred_lower": 0.948395970375896,
          "p_evi_lower": 0.24948706677473803,
          "n_pred_success": 100,
          "n_evi_success": 37,
          "n_samples": 100,
          "alpha": 0.01,
          "theta_stability": 0.5,
          "epsilon_budget": 2,
          "perturbation_space": [
            "abbreviation_swap",
            "ocr_noise",
            "token_insertion",
            "synonym_substitution"
          ],
          "perturbation_counts": {
            "ocr_noise": 23,
            "synonym_substitution": 31,
            "abbreviation_swap": 27,
            "token_insertion": 19
          },
          "reason": "prediction",
          "note_id": "P00327_N00"
        },
        {
          "status": "UNCERTIFIED",
          "prediction_confidence": 0.9280423175983636,
          "provenance_confidence": 0.601434349082817,
          "p_pred_lower": 0.9280423175983636,
          "p_evi_lower": 0.601434349082817,
          "n_pred_success": 99,
          "n_evi_success": 73,
          "n_samples": 100,
          "alpha": 0.01,
          "theta_stability": 0.5,
          "epsilon_budget": 2,
          "perturbation_space": [
            "abbreviation_swap",
            "ocr_noise",
            "token_insertion",
            "synonym_substitution"
          ],
          "perturbation_counts": {
            "synonym_substitution": 19,
            "token_insertion": 23,
            "abbreviation_swap": 26,
            "ocr_noise": 32
          },
          "reason": "prediction",
          "note_id": "P00468_N00"
        },
        {
          "status": "UNCERTIFIED",
          "prediction_confidence": 0.948395970375896,
          "provenance_confidence": 0.49675879666525136,
          "p_pred_lower": 0.948395970375896,
          "p_evi_lower": 0.49675879666525136,
          "n_pred_success": 100,
          "n_evi_success": 63,
          "n_samples": 100,
          "alpha": 0.01,
          "theta_stability": 0.5,
          "epsilon_budget": 2,
          "perturbation_space": [
            "abbreviation_swap",
            "ocr_noise",
            "token_insertion",
            "synonym_substitution"
          ],
          "perturbation_counts": {
            "abbreviation_swap": 30,
            "ocr_noise": 18,
            "synonym_substitution": 32,
            "token_insertion": 20
          },
          "reason": "prediction",
          "note_id": "P00342_N00"
        },
        {
          "status": "UNCERTIFIED",
          "prediction_confidence": 0.948395970375896,
          "provenance_confidence": 0.5906852896007267,
          "p_pred_lower": 0.948395970375896,
          "p_evi_lower": 0.5906852896007267,
          "n_pred_success": 100,
          "n_evi_success": 72,
          "n_samples": 100,
          "alpha": 0.01,
          "theta_stability": 0.5,
          "epsilon_budget": 2,
          "perturbation_space": [
            "abbreviation_swap",
            "ocr_noise",
            "token_insertion",
            "synonym_substitution"
          ],
          "perturbation_counts": {
            "ocr_noise": 24,
            "token_insertion": 23,
            "abbreviation_swap": 24,
            "synonym_substitution": 29
          },
          "reason": "prediction",
          "note_id": "P00416_N00"
        },
        {
          "status": "UNCERTIFIED",
          "prediction_confidence": 0.948395970375896,
          "provenance_confidence": 0.3783974291340614,
          "p_pred_lower": 0.948395970375896,
          "p_evi_lower": 0.3783974291340614,
          "n_pred_success": 100,
          "n_evi_success": 51,
          "n_samples": 100,
          "alpha": 0.01,
          "theta_stability": 0.5,
          "epsilon_budget": 2,
          "perturbation_space": [
            "abbreviation_swap",
            "ocr_noise",
            "token_insertion",
            "synonym_substitution"
          ],
          "perturbation_counts": {
            "token_insertion": 30,
            "abbreviation_swap": 25,
            "synonym_substitution": 20,
            "ocr_noise": 25
          },
          "reason": "prediction",
          "note_id": "P00230_N00"
        },
        {
          "status": "UNCERTIFIED",
          "prediction_confidence": 0.948395970375896,
          "provenance_confidence": 0.8793673007031011,
          "p_pred_lower": 0.948395970375896,
          "p_evi_lower": 0.8793673007031011,
          "n_pred_success": 100,
          "n_evi_success": 96,
          "n_samples": 100,
          "alpha": 0.01,
          "theta_stability": 0.5,
          "epsilon_budget": 2,
          "perturbation_space": [
            "abbreviation_swap",
            "ocr_noise",
            "token_insertion",
            "synonym_substitution"
          ],
          "perturbation_counts": {
            "ocr_noise": 26,
            "synonym_substitution": 25,
            "abbreviation_swap": 23,
            "token_insertion": 26
          },
          "reason": "prediction",
          "note_id": "P00501_N00"
        },
        {
          "status": "UNCERTIFIED",
          "prediction_confidence": 0.948395970375896,
          "provenance_confidence": 0.4564833738464764,
          "p_pred_lower": 0.948395970375896,
          "p_evi_lower": 0.4564833738464764,
          "n_pred_success": 100,
          "n_evi_success": 59,
          "n_samples": 100,
          "alpha": 0.01,
          "theta_stability": 0.5,
          "epsilon_budget": 2,
          "perturbation_space": [
            "abbreviation_swap",
            "ocr_noise",
            "token_insertion",
            "synonym_substitution"
          ],
          "perturbation_counts": {
            "synonym_substitution": 21,
            "token_insertion": 20,
            "abbreviation_swap": 32,
            "ocr_noise": 27
          },
          "reason": "prediction",
          "note_id": "P00617_N00"
        },
        {
          "status": "UNCERTIFIED",
          "prediction_confidence": 0.948395970375896,
          "provenance_confidence": 0.3219272571845858,
          "p_pred_lower": 0.948395970375896,
          "p_evi_lower": 0.3219272571845858,
          "n_pred_success": 100,
          "n_evi_success": 45,
          "n_samples": 100,
          "alpha": 0.01,
          "theta_stability": 0.5,
          "epsilon_budget": 2,
          "perturbation_space": [
            "abbreviation_swap",
            "ocr_noise",
            "token_insertion",
            "synonym_substitution"
          ],
          "perturbation_counts": {
            "abbreviation_swap": 30,
            "ocr_noise": 25,
            "token_insertion": 23,
            "synonym_substitution": 22
          },
          "reason": "prediction",
          "note_id": "P00277_N00"
        },
        {
          "status": "UNCERTIFIED",
          "prediction_confidence": 0.6451489444766687,
          "provenance_confidence": 0.6451489444766687,
          "p_pred_lower": 0.6451489444766687,
          "p_evi_lower": 0.6451489444766687,
          "n_pred_success": 77,
          "n_evi_success": 77,
          "n_samples": 100,
          "alpha": 0.01,
          "theta_stability": 0.5,
          "epsilon_budget": 2,
          "perturbation_space": [
            "abbreviation_swap",
            "ocr_noise",
            "token_insertion",
            "synonym_substitution"
          ],
          "perturbation_counts": {
            "token_insertion": 33,
            "synonym_substitution": 23,
            "abbreviation_swap": 28,
            "ocr_noise": 16
          },
          "reason": "prediction",
          "note_id": "P00139_N00"
        },
        {
          "status": "UNCERTIFIED",
          "prediction_confidence": 0.948395970375896,
          "provenance_confidence": 0.6341071567292269,
          "p_pred_lower": 0.948395970375896,
          "p_evi_lower": 0.6341071567292269,
          "n_pred_success": 100,
          "n_evi_success": 76,
          "n_samples": 100,
          "alpha": 0.01,
          "theta_stability": 0.5,
          "epsilon_budget": 2,
          "perturbation_space": [
            "abbreviation_swap",
            "ocr_noise",
            "token_insertion",
            "synonym_substitution"
          ],
          "perturbation_counts": {
            "synonym_substitution": 34,
            "ocr_noise": 24,
            "abbreviation_swap": 22,
            "token_insertion": 20
          },
          "reason": "prediction",
          "note_id": "P00843_N00"
        },
        {
          "status": "UNCERTIFIED",
          "prediction_confidence": 0.948395970375896,
          "provenance_confidence": 0.6341071567292269,
          "p_pred_lower": 0.948395970375896,
          "p_evi_lower": 0.6341071567292269,
          "n_pred_success": 100,
          "n_evi_success": 76,
          "n_samples": 100,
          "alpha": 0.01,
          "theta_stability": 0.5,
          "epsilon_budget": 2,
          "perturbation_space": [
            "abbreviation_swap",
            "ocr_noise",
            "token_insertion",
            "synonym_substitution"
          ],
          "perturbation_counts": {
            "synonym_substitution": 27,
            "abbreviation_swap": 25,
            "ocr_noise": 23,
            "token_insertion": 25
          },
          "reason": "prediction",
          "note_id": "P00442_N00"
        },
        {
          "status": "UNCERTIFIED",
          "prediction_confidence": 0.948395970375896,
          "provenance_confidence": 0.3879830113670003,
          "p_pred_lower": 0.948395970375896,
          "p_evi_lower": 0.3879830113670003,
          "n_pred_success": 100,
          "n_evi_success": 52,
          "n_samples": 100,
          "alpha": 0.01,
          "theta_stability": 0.5,
          "epsilon_budget": 2,
          "perturbation_space": [
            "abbreviation_swap",
            "ocr_noise",
            "token_insertion",
            "synonym_substitution"
          ],
          "perturbation_counts": {
            "token_insertion": 29,
            "synonym_substitution": 21,
            "abbreviation_swap": 21,
            "ocr_noise": 29
          },
          "reason": "prediction",
          "note_id": "P00193_N00"
        },
        {
          "status": "UNCERTIFIED",
          "prediction_confidence": 0.948395970375896,
          "provenance_confidence": 0.3035051404309871,
          "p_pred_lower": 0.948395970375896,
          "p_evi_lower": 0.3035051404309871,
          "n_pred_success": 100,
          "n_evi_success": 43,
          "n_samples": 100,
          "alpha": 0.01,
          "theta_stability": 0.5,
          "epsilon_budget": 2,
          "perturbation_space": [
            "abbreviation_swap",
            "ocr_noise",
            "token_insertion",
            "synonym_substitution"
          ],
          "perturbation_counts": {
            "abbreviation_swap": 20,
            "ocr_noise": 25,
            "token_insertion": 29,
            "synonym_substitution": 26
          },
          "reason": "prediction",
          "note_id": "P00081_N00"
        },
        {
          "status": "UNCERTIFIED",
          "prediction_confidence": 0.8238856452077388,
          "provenance_confidence": 0.49675879666525136,
          "p_pred_lower": 0.8238856452077388,
          "p_evi_lower": 0.49675879666525136,
          "n_pred_success": 92,
          "n_evi_success": 63,
          "n_samples": 100,
          "alpha": 0.01,
          "theta_stability": 0.5,
          "epsilon_budget": 2,
          "perturbation_space": [
            "abbreviation_swap",
            "ocr_noise",
            "token_insertion",
            "synonym_substitution"
          ],
          "perturbation_counts": {
            "ocr_noise": 26,
            "synonym_substitution": 27,
            "abbreviation_swap": 25,
            "token_insertion": 22
          },
          "reason": "prediction",
          "note_id": "P00857_N00"
        },
        {
          "status": "UNCERTIFIED",
          "prediction_confidence": 0.948395970375896,
          "provenance_confidence": 0.7854533018791034,
          "p_pred_lower": 0.948395970375896,
          "p_evi_lower": 0.7854533018791034,
          "n_pred_success": 100,
          "n_evi_success": 89,
          "n_samples": 100,
          "alpha": 0.01,
          "theta_stability": 0.5,
          "epsilon_budget": 2,
          "perturbation_space": [
            "abbreviation_swap",
            "ocr_noise",
            "token_insertion",
            "synonym_substitution"
          ],
          "perturbation_counts": {
            "synonym_substitution": 17,
            "ocr_noise": 36,
            "token_insertion": 27,
            "abbreviation_swap": 20
          },
          "reason": "prediction",
          "note_id": "P00179_N00"
        },
        {
          "status": "UNCERTIFIED",
          "prediction_confidence": 0.948395970375896,
          "provenance_confidence": 0.6122526884437338,
          "p_pred_lower": 0.948395970375896,
          "p_evi_lower": 0.6122526884437338,
          "n_pred_success": 100,
          "n_evi_success": 74,
          "n_samples": 100,
          "alpha": 0.01,
          "theta_stability": 0.5,
          "epsilon_budget": 2,
          "perturbation_space": [
            "abbreviation_swap",
            "ocr_noise",
            "token_insertion",
            "synonym_substitution"
          ],
          "perturbation_counts": {
            "ocr_noise": 28,
            "abbreviation_swap": 21,
            "token_insertion": 20,
            "synonym_substitution": 31
          },
          "reason": "prediction",
          "note_id": "P00273_N00"
        },
        {
          "status": "UNCERTIFIED",
          "prediction_confidence": 0.9105693270134086,
          "provenance_confidence": 0.8945187462749218,
          "p_pred_lower": 0.9105693270134086,
          "p_evi_lower": 0.8945187462749218,
          "n_pred_success": 98,
          "n_evi_success": 97,
          "n_samples": 100,
          "alpha": 0.01,
          "theta_stability": 0.5,
          "epsilon_budget": 2,
          "perturbation_space": [
            "abbreviation_swap",
            "ocr_noise",
            "token_insertion",
            "synonym_substitution"
          ],
          "perturbation_counts": {
            "token_insertion": 24,
            "ocr_noise": 29,
            "abbreviation_swap": 23,
            "synonym_substitution": 24
          },
          "reason": "prediction",
          "note_id": "P00687_N00"
        },
        {
          "status": "UNCERTIFIED",
          "prediction_confidence": 0.948395970375896,
          "provenance_confidence": 0.3976182462332497,
          "p_pred_lower": 0.948395970375896,
          "p_evi_lower": 0.3976182462332497,
          "n_pred_success": 100,
          "n_evi_success": 53,
          "n_samples": 100,
          "alpha": 0.01,
          "theta_stability": 0.5,
          "epsilon_budget": 2,
          "perturbation_space": [
            "abbreviation_swap",
            "ocr_noise",
            "token_insertion",
            "synonym_substitution"
          ],
          "perturbation_counts": {
            "abbreviation_swap": 23,
            "ocr_noise": 27,
            "synonym_substitution": 31,
            "token_insertion": 19
          },
          "reason": "prediction",
          "note_id": "P00653_N00"
        },
        {
          "status": "UNCERTIFIED",
          "prediction_confidence": 0.9105693270134086,
          "provenance_confidence": 0.35937503229416934,
          "p_pred_lower": 0.9105693270134086,
          "p_evi_lower": 0.35937503229416934,
          "n_pred_success": 98,
          "n_evi_success": 49,
          "n_samples": 100,
          "alpha": 0.01,
          "theta_stability": 0.5,
          "epsilon_budget": 2,
          "perturbation_space": [
            "abbreviation_swap",
            "ocr_noise",
            "token_insertion",
            "synonym_substitution"
          ],
          "perturbation_counts": {
            "ocr_noise": 27,
            "token_insertion": 32,
            "abbreviation_swap": 18,
            "synonym_substitution": 23
          },
          "reason": "prediction",
          "note_id": "P00196_N00"
        }
      ]
    },
    "provenance_first": {
      "n_examples": 30,
      "n_certified": 0,
      "n_base_abstained": 28,
      "certification_rate": 0.0,
      "mean_prediction_confidence": 0.04166633480718015,
      "mean_provenance_confidence": 0.03311725311101676,
      "per_example": [
        {
          "status": "UNCERTIFIED",
          "reason": "base_input_abstained",
          "prediction_confidence": 0.0,
          "provenance_confidence": 0.0,
          "p_pred_lower": 0.0,
          "p_evi_lower": 0.0,
          "note_id": "P00811_N00"
        },
        {
          "status": "UNCERTIFIED",
          "reason": "base_input_abstained",
          "prediction_confidence": 0.0,
          "provenance_confidence": 0.0,
          "p_pred_lower": 0.0,
          "p_evi_lower": 0.0,
          "note_id": "P00415_N00"
        },
        {
          "status": "UNCERTIFIED",
          "reason": "base_input_abstained",
          "prediction_confidence": 0.0,
          "provenance_confidence": 0.0,
          "p_pred_lower": 0.0,
          "p_evi_lower": 0.0,
          "note_id": "P00214_N00"
        },
        {
          "status": "UNCERTIFIED",
          "reason": "base_input_abstained",
          "prediction_confidence": 0.0,
          "provenance_confidence": 0.0,
          "p_pred_lower": 0.0,
          "p_evi_lower": 0.0,
          "note_id": "P00774_N00"
        },
        {
          "status": "UNCERTIFIED",
          "prediction_confidence": 0.7016490846147196,
          "provenance_confidence": 0.49675879666525136,
          "p_pred_lower": 0.7016490846147196,
          "p_evi_lower": 0.49675879666525136,
          "n_pred_success": 82,
          "n_evi_success": 63,
          "n_samples": 100,
          "alpha": 0.01,
          "theta_stability": 0.5,
          "epsilon_budget": 2,
          "perturbation_space": [
            "abbreviation_swap",
            "ocr_noise",
            "token_insertion",
            "synonym_substitution"
          ],
          "perturbation_counts": {
            "synonym_substitution": 27,
            "ocr_noise": 22,
            "token_insertion": 32,
            "abbreviation_swap": 19
          },
          "reason": "prediction",
          "note_id": "P00078_N00"
        },
        {
          "status": "UNCERTIFIED",
          "reason": "base_input_abstained",
          "prediction_confidence": 0.0,
          "provenance_confidence": 0.0,
          "p_pred_lower": 0.0,
          "p_evi_lower": 0.0,
          "note_id": "P00721_N00"
        },
        {
          "status": "UNCERTIFIED",
          "reason": "base_input_abstained",
          "prediction_confidence": 0.0,
          "provenance_confidence": 0.0,
          "p_pred_lower": 0.0,
          "p_evi_lower": 0.0,
          "note_id": "P00247_N00"
        },
        {
          "status": "UNCERTIFIED",
          "reason": "base_input_abstained",
          "prediction_confidence": 0.0,
          "provenance_confidence": 0.0,
          "p_pred_lower": 0.0,
          "p_evi_lower": 0.0,
          "note_id": "P00594_N00"
        },
        {
          "status": "UNCERTIFIED",
          "reason": "base_input_abstained",
          "prediction_confidence": 0.0,
          "provenance_confidence": 0.0,
          "p_pred_lower": 0.0,
          "p_evi_lower": 0.0,
          "note_id": "P00488_N00"
        },
        {
          "status": "UNCERTIFIED",
          "reason": "base_input_abstained",
          "prediction_confidence": 0.0,
          "provenance_confidence": 0.0,
          "p_pred_lower": 0.0,
          "p_evi_lower": 0.0,
          "note_id": "P00461_N00"
        },
        {
          "status": "UNCERTIFIED",
          "reason": "base_input_abstained",
          "prediction_confidence": 0.0,
          "provenance_confidence": 0.0,
          "p_pred_lower": 0.0,
          "p_evi_lower": 0.0,
          "note_id": "P00910_N00"
        },
        {
          "status": "UNCERTIFIED",
          "reason": "base_input_abstained",
          "prediction_confidence": 0.0,
          "provenance_confidence": 0.0,
          "p_pred_lower": 0.0,
          "p_evi_lower": 0.0,
          "note_id": "P00327_N00"
        },
        {
          "status": "UNCERTIFIED",
          "reason": "base_input_abstained",
          "prediction_confidence": 0.0,
          "provenance_confidence": 0.0,
          "p_pred_lower": 0.0,
          "p_evi_lower": 0.0,
          "note_id": "P00468_N00"
        },
        {
          "status": "UNCERTIFIED",
          "reason": "base_input_abstained",
          "prediction_confidence": 0.0,
          "provenance_confidence": 0.0,
          "p_pred_lower": 0.0,
          "p_evi_lower": 0.0,
          "note_id": "P00342_N00"
        },
        {
          "status": "UNCERTIFIED",
          "prediction_confidence": 0.5483409596006847,
          "provenance_confidence": 0.49675879666525136,
          "p_pred_lower": 0.5483409596006847,
          "p_evi_lower": 0.49675879666525136,
          "n_pred_success": 68,
          "n_evi_success": 63,
          "n_samples": 100,
          "alpha": 0.01,
          "theta_stability": 0.5,
          "epsilon_budget": 2,
          "perturbation_space": [
            "abbreviation_swap",
            "ocr_noise",
            "token_insertion",
            "synonym_substitution"
          ],
          "perturbation_counts": {
            "abbreviation_swap": 26,
            "token_insertion": 27,
            "synonym_substitution": 27,
            "ocr_noise": 20
          },
          "reason": "prediction",
          "note_id": "P00416_N00"
        },
        {
          "status": "UNCERTIFIED",
          "reason": "base_input_abstained",
          "prediction_confidence": 0.0,
          "provenance_confidence": 0.0,
          "p_pred_lower": 0.0,
          "p_evi_lower": 0.0,
          "note_id": "P00230_N00"
        },
        {
          "status": "UNCERTIFIED",
          "reason": "base_input_abstained",
          "prediction_confidence": 0.0,
          "provenance_confidence": 0.0,
          "p_pred_lower": 0.0,
          "p_evi_lower": 0.0,
          "note_id": "P00501_N00"
        },
        {
          "status": "UNCERTIFIED",
          "reason": "base_input_abstained",
          "prediction_confidence": 0.0,
          "provenance_confidence": 0.0,
          "p_pred_lower": 0.0,
          "p_evi_lower": 0.0,
          "note_id": "P00617_N00"
        },
        {
          "status": "UNCERTIFIED",
          "reason": "base_input_abstained",
          "prediction_confidence": 0.0,
          "provenance_confidence": 0.0,
          "p_pred_lower": 0.0,
          "p_evi_lower": 0.0,
          "note_id": "P00277_N00"
        },
        {
          "status": "UNCERTIFIED",
          "reason": "base_input_abstained",
          "prediction_confidence": 0.0,
          "provenance_confidence": 0.0,
          "p_pred_lower": 0.0,
          "p_evi_lower": 0.0,
          "note_id": "P00139_N00"
        },
        {
          "status": "UNCERTIFIED",
          "reason": "base_input_abstained",
          "prediction_confidence": 0.0,
          "provenance_confidence": 0.0,
          "p_pred_lower": 0.0,
          "p_evi_lower": 0.0,
          "note_id": "P00843_N00"
        },
        {
          "status": "UNCERTIFIED",
          "reason": "base_input_abstained",
          "prediction_confidence": 0.0,
          "provenance_confidence": 0.0,
          "p_pred_lower": 0.0,
          "p_evi_lower": 0.0,
          "note_id": "P00442_N00"
        },
        {
          "status": "UNCERTIFIED",
          "reason": "base_input_abstained",
          "prediction_confidence": 0.0,
          "provenance_confidence": 0.0,
          "p_pred_lower": 0.0,
          "p_evi_lower": 0.0,
          "note_id": "P00193_N00"
        },
        {
          "status": "UNCERTIFIED",
          "reason": "base_input_abstained",
          "prediction_confidence": 0.0,
          "provenance_confidence": 0.0,
          "p_pred_lower": 0.0,
          "p_evi_lower": 0.0,
          "note_id": "P00081_N00"
        },
        {
          "status": "UNCERTIFIED",
          "reason": "base_input_abstained",
          "prediction_confidence": 0.0,
          "provenance_confidence": 0.0,
          "p_pred_lower": 0.0,
          "p_evi_lower": 0.0,
          "note_id": "P00857_N00"
        },
        {
          "status": "UNCERTIFIED",
          "reason": "base_input_abstained",
          "prediction_confidence": 0.0,
          "provenance_confidence": 0.0,
          "p_pred_lower": 0.0,
          "p_evi_lower": 0.0,
          "note_id": "P00179_N00"
        },
        {
          "status": "UNCERTIFIED",
          "reason": "base_input_abstained",
          "prediction_confidence": 0.0,
          "provenance_confidence": 0.0,
          "p_pred_lower": 0.0,
          "p_evi_lower": 0.0,
          "note_id": "P00273_N00"
        },
        {
          "status": "UNCERTIFIED",
          "reason": "base_input_abstained",
          "prediction_confidence": 0.0,
          "provenance_confidence": 0.0,
          "p_pred_lower": 0.0,
          "p_evi_lower": 0.0,
          "note_id": "P00687_N00"
        },
        {
          "status": "UNCERTIFIED",
          "reason": "base_input_abstained",
          "prediction_confidence": 0.0,
          "provenance_confidence": 0.0,
          "p_pred_lower": 0.0,
          "p_evi_lower": 0.0,
          "note_id": "P00653_N00"
        },
        {
          "status": "UNCERTIFIED",
          "reason": "base_input_abstained",
          "prediction_confidence": 0.0,
          "provenance_confidence": 0.0,
          "p_pred_lower": 0.0,
          "p_evi_lower": 0.0,
          "note_id": "P00196_N00"
        }
      ]
    }
  },
  "knowledge_graph_summary": {
    "baseline": {
      "site_id": "bert_site_baseline",
      "n_total_edges": 150,
      "n_abstained": 0,
      "abstention_rate": 0.0,
      "relation_type_counts": {
        "REL_IDX_0": 79,
        "REL_IDX_2": 49,
        "REL_IDX_1": 22
      },
      "mean_coverage": 0.03187447263548771
    },
    "provenance_first": {
      "site_id": "bert_site_provenance_first",
      "n_total_edges": 150,
      "n_abstained": 144,
      "abstention_rate": 0.96,
      "relation_type_counts": {
        "REL_IDX_1": 6
      },
      "mean_coverage": 0.10367791230479877
    }
  }
}
