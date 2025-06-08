import yaml, argparse, warnings, os
import pandas as pd
from quantsys.data_loader import load_all_frames
from quantsys.labeller     import label_future_move
from quantsys.features.base import build_feature_df
from quantsys.miners.rule_miner import mine_rules
from quantsys.miners.lgbm_model import run_lgbm
from quantsys.miners.tft_model  import run_tft
from quantsys.backtester        import walk_forward_backtest

warnings.filterwarnings("ignore")

def main(cfg):
    # 1️⃣ Load & unify
    raw = load_all_frames(cfg)
    # 2️⃣ Label
    labelled = label_future_move(raw, cfg)
    # 3️⃣ Feature matrix
    feats = build_feature_df(labelled, cfg)
    
    # 4️⃣ Pattern mining 🚀
    rule_report = mine_rules(feats, cfg)
    lgbm_report = run_lgbm(feats, cfg)
    tft_report  = run_tft(feats, cfg)
    
    # 5️⃣ Back-test on each engine
    results = {
        "rules" : walk_forward_backtest(rule_report["signals"], labelled),
        "lgbm"  : walk_forward_backtest(lgbm_report ["signals"], labelled),
        "tft"   : walk_forward_backtest(tft_report  ["signals"], labelled),
    }
    for k,v in results.items():
        print(f"\n=== {k.upper()} SUMMARY ===")
        print(v["metrics"].to_markdown())
        v["equity_curve"].plot(title=f"{k.upper()} equity")
    
    # save artefacts
    os.makedirs("artefacts", exist_ok=True)
    feats.to_parquet("artefacts/feature_matrix.parquet")
    pd.to_pickle(results, "artefacts/backtest_results.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--config", default="config.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg)
