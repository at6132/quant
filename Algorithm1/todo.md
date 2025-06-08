Load your five parquet files (15 sec, 1 min, 15 min, 1 h, 4 h) that already include all your indicator columns.

Clean them up

Make sure every candle has a proper UTC timestamp, no gaps, no NaNs.

Rename/standardise columns so they line up across files.

Blend the time-frames together

For each 15-second candle we attach “what the 1 min / 15 min / 1 h / 4 h chart was saying at that moment” (e.g., higher-TF trend flags, SMC/BOS tags, etc.).

Create extra features – rolling stats, time-since-event counters, cross-indicator combos, whatever looks useful.

Label the rows

Look ahead a fixed horizon (say 30 minutes).

If price moves +$500 before it drops −$500 → label = LongSuccess.

If price drops −$500 first → label = ShortSuccess (or LongFail).

Save the move size and how long it took.

Hunt for patterns in those features that consistently show up before the labelled moves.

Rule search: market-structure style “if-this-and-that” combos.

ML model: feed everything into LightGBM to let it discover non-obvious combos.

Back-test the signals you get from either method, including costs/slippage, and spit out equity curves & stats.

Report the findings in an easy-to-read notebook/dashboard (feature importance charts, confusion matrix, P&L curve).

Nothing here trades live; it just chews through the historical parquet data and tells you “these conditions tended to come just before $500 moves – and this is how profitable they’d have been.”