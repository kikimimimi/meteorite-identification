# Inferred Test Labels From Submission Comparisons

This file records image-level label hypotheses inferred from Kaggle public-score changes when two submissions differ by only a few images.

Important:

- These are not ground-truth labels.
- They are public-leaderboard inferences.
- A higher score after swapping images suggests the swapped-in set is better than the swapped-out set, but it does not always prove every individual image label.
- Use this file to guide boundary re-ranking and manual review around `top105`.

Label convention:

```text
1 = likely meteorite / positive
0 = likely non-meteorite / negative
```

## Strong Evidence

| id | inferred_label | confidence | evidence |
|---|---:|---|---|
| `000173.jpg` | 1 | strong | `submission_original_inr_w040_top105.csv` improved over the original baseline by swapping in `000173.jpg` and removing `000083.jpg`; xstrong later dropped after replacing `000173.jpg` with `000074.jpg`. |
| `000083.jpg` | 0 | strong | Original strong baseline improved when `000083.jpg` was removed and `000173.jpg` was added. |
| `000074.jpg` | 0 | strong | xstrong INR replaced `000173.jpg` with `000074.jpg` and public score decreased; SVM INR also added `000074.jpg` and scored lower than LogReg INR. |

## Medium Evidence

| id | inferred_label | confidence | evidence |
|---|---:|---|---|
| `000180.jpg` | 1 | medium | Strong INR LogReg kept `000180.jpg`; strong INR SVM removed it and added `000098.jpg`, and SVM score dropped from about `0.764` to `0.75`. |
| `000098.jpg` | 0 | medium | Strong INR SVM added `000098.jpg` while removing `000180.jpg`; SVM score dropped. However, LP variants sometimes selected `000098.jpg`, so mark as medium rather than strong. |
| `000097.jpg` | 0 | medium | Strong INR `top106` added `000097.jpg` beyond the `top105` set and score decreased from `0.76439` to `0.76041`, suggesting the extra sample is likely negative or not worth including. |
| `000016.jpg` | 0 | medium | Strong INR `top107` added `000097.jpg` and `000016.jpg` beyond the `top105` set; score decreased further to `0.75647`, matching the expected F1 drop when both added images are negatives. |
| `000017.jpg` | 1 | medium | LP+INR improved over LP-only by adding `000017.jpg`, `000084.jpg`, `000174.jpg` and removing `000021.jpg`, `000045.jpg`, `000179.jpg`. |
| `000084.jpg` | 1 | medium | Same LP+INR improvement set as above; also selected by INR-only variants. |
| `000174.jpg` | 1 | medium | Same LP+INR improvement set as above; also selected by INR-only variants. |
| `000069.jpg` | 1 | medium | DINOv2 LoRA quick `top106` improved from `0.79581` at top105 to `0.80208`; the added rank-106 image was `000069.jpg`, suggesting it is positive. |
| `000021.jpg` | 0 | medium | LP+INR improved over LP-only while removing `000021.jpg`. |
| `000179.jpg` | 0 | medium | LP+INR improved over LP-only while removing `000179.jpg`. |

## Weak Or Conflicting Evidence

| id | inferred_label | confidence | evidence |
|---|---:|---|---|
| `000045.jpg` | 0 | weak/conflicting | LP+INR improved over LP-only while removing `000045.jpg`, but INR-only variants selected it in some comparisons. |
| `000056.jpg` | unknown | conflicting | Selected by INR-only over LP variants, but no direct score-isolating comparison proves it. |
| `000126.jpg` | unknown | conflicting | Selected by INR-only over LP variants, but no direct score-isolating comparison proves it. |
| `000127.jpg` | unknown | conflicting | Selected by INR-only over LP variants, but no direct score-isolating comparison proves it. |
| `000048.jpg` | unknown | weak | Selected by LP/LP+INR over INR-only in the three-way comparison, but no direct public-score isolation. |
| `000092.jpg` | unknown | weak | Selected by LP/LP+INR over INR-only in the three-way comparison, but no direct public-score isolation. |
| `000101.jpg` | unknown | weak | Selected by LP/LP+INR over INR-only in the three-way comparison, but no direct public-score isolation. |

## Neutral / No Reliable Inference

These comparisons changed labels but produced the same public score, so they do not provide useful individual-label evidence:

- `submission_lp_no_stage1_top105.csv` vs `submission_all_methods_top105.csv` had the same score and differed on:
  - `000020.jpg`
  - `000101.jpg`
  - `000151.jpg`
  - `000156.jpg`
- `submission_original_inr_w060_top105.csv` and `submission_original_inr_w040_top105.csv` were identical.
- `submission_original_inr_strong_w045_top105.csv` and `submission_original_inr_strong_w040_top105.csv` were identical.

## Current Practical Rules

For boundary re-ranking around `top105`:

```text
Prefer keeping: 000173.jpg, 000180.jpg, 000069.jpg
Prefer excluding: 000083.jpg, 000074.jpg, 000097.jpg, 000016.jpg
Be cautious with: 000098.jpg, 000045.jpg, 000056.jpg, 000126.jpg, 000127.jpg
```

If a future candidate submission differs only by these ids, prioritize the direction above unless a new public-score comparison contradicts it.
