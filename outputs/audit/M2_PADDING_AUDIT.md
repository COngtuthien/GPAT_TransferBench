# M2 — Crop Padding Audit (Q-22 / DEV-016)

Counts of detected crops whose requested 1.25x square extends past a frame edge and is therefore zero-padded on that side. The requested square is never shrunk or shifted, so a padded crop is still a true square before the 256 resize.

**The crop policy is frozen and is not changed because of these frequencies.**

| dataset | detected crops | no padding | single side | multiple sides | left | right | top | bottom |
|---|---|---|---|---|---|---|---|---|
| msu_mfsd | 2,240 | 2,209 | 31 | 0 | 0 | 0 | 6 | 25 |
| siwmv2 | 13,575 | 11,768 | 1,197 | 610 | 680 | 638 | 543 | 785 |

CASIA-FASD performs no SCRFD crop, so it contributes no row here.

## Most extreme padded samples (preserved for audit)

| pad fraction of side | sample_id | dataset | side_px | pads L,T,R,B |
|---|---|---|---|---|
| 0.5991 | `c0888264b24120f0…` | siwmv2 | 1,282 | [262, 0, 300, 206] |
| 0.5906 | `a6190049596f9ed8…` | siwmv2 | 1,270 | [250, 0, 300, 200] |
| 0.5816 | `7977cbd8b49911e4…` | siwmv2 | 1,269 | [250, 3, 299, 186] |
| 0.5816 | `1a91c43abec8a44a…` | siwmv2 | 1,269 | [249, 2, 300, 187] |
| 0.5620 | `890e5d7faae73137…` | siwmv2 | 1,121 | [224, 0, 177, 229] |
| 0.5619 | `4d4802b7163bf70f…` | siwmv2 | 1,123 | [224, 0, 179, 228] |
| 0.5566 | `ba9c39930db7cb6d…` | siwmv2 | 1,121 | [225, 0, 176, 223] |
| 0.5560 | `ed045285089e4436…` | siwmv2 | 1,117 | [224, 0, 173, 224] |
| 0.5557 | `e8d3bfd964580243…` | siwmv2 | 1,123 | [222, 0, 181, 221] |
| 0.5460 | `a7c4eb633e1f0ac1…` | siwmv2 | 1,108 | [221, 0, 167, 217] |
