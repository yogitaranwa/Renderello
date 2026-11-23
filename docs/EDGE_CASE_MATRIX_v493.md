# Product-photo edge cases and actual coverage

Status refers to this version, not future plans. Intake is opt-in and does not repair semantic product failures.

| Edge case | Current handling | What is not established |
|---|---|---|
| EXIF 1-8 | tested derivative orientation | owner-PC viewer differences |
| Invalid EXIF orientation | refused | automatic repair |
| RGB with no ICC | explicit sRGB assumption warning | calibrated colour accuracy |
| Embedded ICC | explicit conversion policy, tested sRGB fixture | every vendor profile |
| Broken ICC | refused, no silent fallback | profile repair |
| Unprofiled CMYK | refused | guessed colour conversion |
| 16-bit PNG including RGB16 | header refusal tested | tone mapping |
| High-bit-depth TIFF / float | refused | HDR rendering |
| RGBA / LA / palette transparency | retained, tested native alpha | semantic mask correctness |
| Animated / multipage | refused | choosing an intended frame |
| Empty / random / corrupt / truncated | refused | reconstruction |
| Encoded-byte / side / pixel budgets | explicit limits tested | arbitrary resource exhaustion attacks |
| Extreme aspect | policy refusal or warning | template framing quality |
| Low resolution | retained, warning | recovered detail or readable print |
| Unsupported formats | refused | HEIC/AVIF/RAW ingestion |
| FIFO / nonregular file | refused before blocking open | network filesystem reliability |
| Repeated/concurrent output path | no overwrite, staged failure evidence | transactional multi-file commit |
| Fine print / labels | full-resolution anonymous crops | semantic OCR correctness or exact generated text |
| Glass / transparent bodies | alpha retained, review crops | refraction or missing glass reconstruction |
| Reflective metal / polished packaging | review only | physically correct reflections |
| Fur / hair / soft boundaries | review only | clean semitransparent masks |
| White-on-white / black objects | review only | automatic segmentation success |
| Multiple products | review only | all-object mask completeness |
| Source-cropped product | refusal to infer completion claims | restoration of unseen pixels |
| Existing source shadows | review only | separation from new shadows |
| Failed generation | explicit review entry | a scored quality result |
| Duplicate output submitted twice | content-based rejection tested | independence of methods |
| Candidate shape mismatch | refused, no silent resizing | geometric registration |
| Rater identity | not inferred; CSV blank | independent evaluator authentication |
| Review content changed | integrity rejection tested | tamper-proof signatures |

A visually bad generation must still fail publication review even when every software test passes.
