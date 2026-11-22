# Which district matters for deciding whether a tool can answer?

Analysis plan, November 21, 2026 (New York). Written before collecting model outcomes for this prompt bank. Earlier TARA results motivated the design. This local file is a time-stamped plan, without independent registration.

## Question

When a tool description mentions two districts, can Qwen identify which district governs whether it can answer a request? Does the activation at that district word carry more information used by the decision than the same word in a field about the office that wrote the documentation?

## Design

Use the original Qwen2.5-1.5B-Instruct checkpoint, revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306. Create eight read-only fictitious public information functions. Each user asks about one district. Independently toggle whether the named service district matches the request and whether the documentation office district matches. The two fields use identical district names and change one token at a time. Each tool has four requests targeting East, West, North and South. East pairs with West, North with South. Reverse field order across tools and requests in a checkerboard. Present a tool card as native schema or inline JSON. Use the same A/B metalinguistic applicability question in both. Reverse what A and B mean. These 8 × 4 × 4 × 2 × 2 = 512 baseline prompts are the full bank. Raw prompts and token positions must be saved. No tools are executed.

## Measurement

For each prompt, score the first-token logit for the letter meaning applicable minus the logit for the letter meaning inapplicable. These are log-odds within the two-letter choice. Also save the unconstrained highest-logit token and total probability assigned to A or B to show when a two-letter readout misrepresents free output. The readout is a judgment about a tool. It is a different endpoint from emitting or executing a function call.

The primary behavioral contrast is the change in this margin from matching the service district minus the change from matching the office district. Average over the other field, label mappings, four directions, and both formats inside each tool family, then average the eight families equally. Report all family means, both individual field effects, and correct two-letter predictions. Contrast formats descriptively only.

For each recipient, overwrite the output of decoder block 14 (one-based, midway through 28 blocks) at the one changed district token with the activation of its single-field-flipped counterpart. Run both directions for service and office, leaving later positions untouched at the intervention. Orient the change so a positive value means that inserting the requested district increases the applicable margin. Primary intervention contrast is mean oriented service patch minus mean oriented office patch. This tests influence at one position and depth. Other positions may already contain the relevant signal. No layer search or model fitting.

Check the tokenizer differs at exactly one index in each matched counterfactual. For the first requested district, inline JSON, A=applicable configuration in each family, check identity patch at mid-depth and the corresponding embedding-level substitutions for both words. At embedding depth, replacing the sole differing token should reproduce its donor's margin. Reject numerical checks with an absolute error above 0.05 logits. Check all expected cells for completeness, duplicate keys and finite scores.

Uncertainty is variation among these eight deliberately constructed tool families. Report the primary mean after leaving each family out, and the min/max family effects. No population p-value or clinical application estimate. The main rival explanations include schema-field order, A/B label preference, weak free compliance and fixed vocabulary. Counterbalancing limits these explanations within the bank. Transfer to new layouts, models, free calls and naturally authored documentation is an open question. A null or negative control result is retained.
