# M8 learned-bank S7 readiness summary (v2)

**Status:** `ALL_FOUR_LEARNED_METHODS_S7_QUALIFIED`. The earlier summary (E05 blocked) is kept unchanged as history.

| Method | S7 record | Gates | Layout | Bitwise A/B (3 seeds) | Readiness |
|---|---|---|---|---|---|
| E06c | `LEARNED_BANK_S7_E06c.json` (`8fb5bac`) | 98/98 | single_process_three_seeds | identical | READY_FOR_SEPARATELY_AUTHORIZED_PRODUCTION |
| E05 | `LEARNED_BANK_S7_E05.json` (`1e65064`) | 145/145 | per_seed_process | identical | E05_S7_QUALIFIED_PRODUCTION_AUTHORITY_PENDING |
| E03 | `LEARNED_BANK_S7_E03.json` (`cd9c294`) | 98/98 | single_process_three_seeds | identical | READY_FOR_SEPARATELY_AUTHORIZED_PRODUCTION |
| E06b | `LEARNED_BANK_S7_E06b.json` (`f84f05f`) | 98/98 | single_process_three_seeds | identical | READY_FOR_SEPARATELY_AUTHORIZED_PRODUCTION |

E05 history: attempt 1 failed on the source-root path binding (`2d430e9`) → O2 ruling (`5f81d70`) → B1/B2 found (`ef96fcb`) → B1 + B2 P2 ruling (`7ff4d36`) → qualification (`d571650`) → attempt 2 PASSED (`1e65064`).

Production needs a separate owner `learned_bank_production_<M>.yaml` per method. Disclosure: the E05 S7 outputs are low-contrast (pixels 95–132, mean per-image std 5–8). This is descriptive, not a gate. No production, downstream, TEST, DINOv3 or GPAT work was done.
