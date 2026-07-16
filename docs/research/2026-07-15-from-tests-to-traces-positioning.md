# From Tests to Traces: Computational Psychometrics for AI-Mediated Cognition

**Status**: Positioning draft v0.1 — internal; source material for academic outreach one-pager, whitepaper preface, and pitch narrative
**Date**: 2026-07-15
**Audience**: academic collaborators (psychology, HCI, education measurement), technical partners
**Related**: JDVP protocol ([BufferLine/JDVP-protocol](https://github.com/BufferLine/JDVP-protocol)), proposal JDVP-2026-002 (conversation-unit measurement), completeness review (2026-07-15)

---

## Thesis

BufferLine builds **computational psychometrics infrastructure**: it transforms conversational traces of human–AI interaction into validated measurements of cognitive constructs — beginning with judgment delegation, the behavioral inverse of critical-thinking disposition in action.

The thesis has three claims:

1. **From tests to traces.** Critical thinking is a *disposition*, and the valid measurement of a disposition is unobtrusive observation of typical performance — not episodic maximum-performance testing. Until now, observation cost made this impossible at scale.
2. **The instrument-cost collapse.** Foundation models turned semantic annotation from artisanal expert labor into commodity compute, with documented annotator-level performance. Intensive longitudinal measurement of cognition from conversation logs is economically feasible for the first time.
3. **Value concentrates in the metrology layer.** As extraction commoditizes, durable value moves to validated construct definitions, calibration anchors, and norms — the measurement standard, not the measurement engine. JDVP is that standard.

A one-line version for a general audience: *a continuous glucose monitor for critical thinking* — episodic lab tests did not lose to the CGM on single-point accuracy; they lost on density and ecological context.

---

## 1. The measurement gap

Human–AI interaction research surged after 2023, and its empirical core is a measurement problem. The two landmark 2025 studies illustrate the gap from opposite ends: Lee et al. (CHI 2025) measured critical-thinking reduction in 319 knowledge workers — via **self-report**, a limitation the authors state explicitly; Kosmyna et al. (MIT Media Lab, 2025) measured "cognitive debt" in essay writing — via **lab EEG**, unscalable by construction. Between self-report (reactive, retrospective, coarse) and neuroimaging (precise, tiny-n, artificial), there is no instrument that measures cognitive engagement with AI **where it actually happens: in the interaction log, continuously, at scale**.

That absent instrument is the product category.

## 2. Claim 1 — From tests to traces

The psychometric argument rests on five established results:

**Dispositions demand typical-performance measurement.** Cronbach's (1949) distinction between maximum and typical performance, empirically sharpened by Sackett, Zedeck & Fogli (1988), separates what a person *can* do under motivated test conditions from what they *habitually* do. The critical-thinking literature itself defines CT as a two-part construct — skills *and* dispositions (Facione's 1990 Delphi consensus; Perkins, Jay & Tishman's dispositional theory). Every major CT instrument (Watson–Glaser, CCTST, HCTA) is a maximum-performance test. The dispositional half of the construct has never had a matching instrument.

**Traces are nonreactive.** Test-taking alters the behavior it measures — reactivity, demand characteristics (Orne, 1962), practice effects. Webb et al. (1966) made the classic case for unobtrusive measures; conversational traces are the modern unobtrusive measure of reasoning behavior.

**Aggregation buys back reliability.** Single behavioral observations are noisy; Epstein (1979) showed that aggregation across occasions stabilizes them dramatically, and the Spearman–Brown relation quantifies the exchange. The trace paradigm deliberately trades per-observation precision for observation density. Our empirical turn-level ceiling (self-consistency r≈0.56 on natural conversation) is not a defect of the approach but its expected regime — at that single-observation reliability, roughly seven independent observations suffice for trait-grade (0.90) aggregate reliability. Density is exactly what traces provide and tests cannot.

**Between-person psychometrics does not license within-person inference.** Molenaar's (2004) ergodicity critique — within-person process structure is not mathematically recoverable from between-person data — motivated the field's turn toward intensive longitudinal designs (experience sampling: Csikszentmihalyi & Larson 1987; EMA: Stone & Shiffman 1994). Conversation logs are the cheapest intensive longitudinal data in existence, and JDVP's design (observed baseline, within-person trends, interaction-level aggregates) is native to this paradigm.

**Institutional precedent exists.** Education measurement already operationalized "assessment without tests": stealth assessment (Shute, 2011), Evidence-Centered Design as its formal claims–evidence–task architecture (Mislevy, Steinberg & Almond, 2003), and process-data psychometrics in PISA log files. JDVP applies the same move to reasoning behavior in AI-mediated decisions.

## 3. Claim 2 — Why LLMs, why now

**Theoretical basis.** The distributional hypothesis — meaning is recoverable from patterns of use (Harris, 1954; Firth, 1957) — passed from static embeddings (Mikolov et al., 2013) through contextual encoders (Devlin et al., 2018) to foundation models. Semantic properties of text, including pragmatic and stance-level properties, are operationalized as positions in learned representation spaces.

**The paradigm break.** Before foundation models, measuring a new construct from text required a supervised dataset *per construct* — months of annotation before the first measurement. In-context learning (Brown et al., 2020; Bommasani et al., 2021) collapsed the marginal cost of operationalizing a new semantic variable to the cost of writing a prompt. This — not raw capability — is what makes construct-level measurement iteration (our v1 → v1.5 → v1.6 cycle) economically possible.

**Validated annotator performance.** A rapidly consolidating literature establishes LLMs at or above human-annotator agreement on text classification: Gilardi, Alizadeh & Kubli (2023, *PNAS*) vs crowd workers; Törnberg (2023) vs experts on political text; Rathje et al. (2024, *PNAS* 121(34)) for multilingual psychological text analysis specifically; Ziems et al. (2024, *Computational Linguistics*) across computational social science tasks, with documented bias caveats. The methodological legitimacy of "LLM as measurement instrument" no longer needs to be argued from scratch — it needs to be *earned per construct* via validation, which is our research agenda.

**The cost curve.** Inference cost at constant capability has fallen by roughly an order of magnitude per year, and open-weight models run on consumer hardware. Our own benchmark instantiates the trend: a $0 local 26B model (gemma-class) matches cloud reference models on the protocol's operative units — delegation-vector correlation 0.53 vs 0.50, trend agreement 65% vs 64% (6 models × 300 conversations × 4,103 turns). The extraction frontier descends the semantic-depth hierarchy as cost falls — and our observability data traces that hierarchy directly: information seeking (behavioral, r=0.63) > judgment holder (0.60) > cognitive engagement (0.55) > delegation awareness (metacognitive, 0.47).

**Measurement discipline imported into ML.** Jacobs & Wallach (2021) brought measurement modeling — the construct/operationalization distinction — into computational systems; the human-label-variation literature (Aroyo & Welty, 2015; Plank, 2022) reframes annotator disagreement as construct ambiguity rather than noise. Both are load-bearing for our design decisions: the r≈0.56 ceiling is documented as a task property, reporting standards are chance-corrected, and disagreement boundaries (rising/stable at |slope|≈0.1) are treated as informative.

## 4. Claim 3 — Value concentrates in the metrology layer

If Claim 2 holds, extraction becomes commodity — anyone will score conversations. What remains scarce:

1. **The ruler**: validated construct definitions, anchored scales, calibration sets, and the social agreement to measure on them. In measurement science, thermometers commoditized; the ITS-90 temperature scale did not.
2. **Norms**: population reference data for interpreting individual measurements — accumulated, not downloaded.
3. **Trust**: validity evidence in the sense of Messick's (1989) unified framework and Kane's (2013) argument-based validation — an explicit chain from observation to interpretation to use, of the kind the AERA/APA/NCME *Standards* require.

JDVP is engineered as that layer: an open descriptive protocol (observation without evaluation), versioned semantics with empirical change control, observer profiles and calibration requirements, and reporting standards (chance-corrected agreement, conversation-level resampled uncertainty, conditioning disclosure). The strategic bet is explicit: **when measurement becomes possible for everyone, the winner is whoever defined the accepted scale.**

## 5. Lineages and precedents

The approach belongs to a sparse but real tradition — "AI as scientific instrument," distinct from the dominant "AI as labor" investment thesis:

- **Text-as-data / computational social science**: Grimmer & Stewart (2013) → LLM-annotation era (above).
- **Language-based assessment**: LIWC (Pennebaker) → open-vocabulary personality and mental-health prediction from social media language (Schwartz et al., 2013; Kosinski, Stillwell & Graepel, 2013, *PNAS*; Eichstaedt et al., 2018, *PNAS*). JDVP is this lineage with the construct moved to judgment delegation and the corpus moved to human–AI dialogue.
- **Digital phenotyping / language biomarkers**: speech-based psychosis prediction (Bedi et al., 2015), clinical-grade validation pathways for trace-based measures.
- **Automated fidelity coding — the closest commercial analog**: motivational-interviewing fidelity, once hand-coded by psychologists, automated and clinically validated (Atkins et al., 2014; commercialized by Lyssn). Expert manual coding → automated measurement → academic validation → institutional adoption is precisely our sequence.
- **Population-level interaction measurement**: large-lab work (e.g., Anthropic's Clio-style usage analyses) measures delegation-adjacent patterns at population level but structurally avoids individual measurement — the reputational economics of frontier labs leave the individual-instrument layer open.

## 6. What we are not claiming

Credibility in this space is earned by boundary discipline:

- **No theory-of-mind claims.** We characterize LLM observers by task-level agreement with human annotators (socio-pragmatic inference performance), not by mentalistic capability claims — that literature is contested and unnecessary for our argument.
- **LLM consensus is not validity.** Shared training data can produce shared bias; our validation chain currently bottoms out in LLM agreement, and we treat this as the top-priority gap. The near-term research agenda is human-anchored: gold-standard annotation with human–human reliability reported, convergent validity against established instruments (Weight of Advice, Need for Cognition, MAI, CT tests), and self-report micro-prompts as an EMA-style criterion channel.
- **Measurement is not surrogacy.** We measure human traces; we do not simulate human participants (cf. Messeri & Crockett, 2024, *Nature*, on AI-driven illusions of understanding in research).
- **Description is not evaluation.** The protocol layer is strictly non-normative — no scores-as-authority, no rankings, no risk labels. Normative applications (assessment products, certification) live in explicitly separate layers with their own governance, mindful of the regulatory perimeter around scoring practices (e.g., EU AI Act Art. 5).
- **Text has an information ceiling.** Cognition that leaves no trace in the interaction is not measurable from the interaction; hybrid channels (behavioral telemetry, micro self-report) are part of the roadmap, not an afterthought. Structurally, however, the measurable share grows over time: as AI mediates more cognitive work, more cognition passes through recordable interfaces.

## 7. Research agenda (pointer)

The operational roadmap follows from the three claims: (i) conversation-unit measurement standardization (proposal JDVP-2026-002: Interaction Summary, derived observables, reporting standards); (ii) human-anchored validation (gold set, convergent validity study — designed for academic collaboration); (iii) longitudinal trait-estimation standards (test–retest, generalizability decomposition, minimum-observation rules) as the prerequisite for any assessment-layer product.

---

## References (working list)

*Foundations — psychometrics of typical performance and traces*
- Cronbach, L. J. (1949). *Essentials of Psychological Testing.*
- Sackett, P. R., Zedeck, S., & Fogli, L. (1988). Relations between measures of typical and maximum job performance. *Journal of Applied Psychology*, 73(3).
- Webb, E. J., Campbell, D. T., Schwartz, R. D., & Sechrest, L. (1966). *Unobtrusive Measures.*
- Orne, M. T. (1962). On the social psychology of the psychological experiment. *American Psychologist*, 17(11).
- Epstein, S. (1979). The stability of behavior: I. On predicting most of the people much of the time. *JPSP*, 37(7).
- Molenaar, P. C. M. (2004). A manifesto on psychology as idiographic science. *Measurement*, 2(4).
- Csikszentmihalyi, M., & Larson, R. (1987). Validity and reliability of the Experience-Sampling Method. *JNMD*, 175(9).
- Stone, A. A., & Shiffman, S. (1994). Ecological momentary assessment. *Annals of Behavioral Medicine*, 16(3).
- Shute, V. J. (2011). Stealth assessment in computer-based games to support learning.
- Mislevy, R. J., Steinberg, L. S., & Almond, R. G. (2003). On the structure of educational assessments. *Measurement*, 1(1).
- Messick, S. (1989). Validity. In *Educational Measurement* (3rd ed.).
- Kane, M. T. (2013). Validating the interpretations and uses of test scores. *JEM*, 50(1).

*Critical thinking as disposition*
- Facione, P. A. (1990). *Critical Thinking: A Statement of Expert Consensus* (Delphi Report).
- Perkins, D., Jay, E., & Tishman, S. (1993). Beyond abilities: A dispositional theory of thinking. *Merrill-Palmer Quarterly*, 39(1).

*Why LLMs, why now*
- Harris, Z. (1954). Distributional structure. *Word*, 10. / Firth, J. R. (1957). A synopsis of linguistic theory.
- Mikolov, T., et al. (2013). Distributed representations of words and phrases. *NeurIPS*.
- Devlin, J., et al. (2018). BERT. *NAACL 2019*.
- Brown, T., et al. (2020). Language models are few-shot learners. *NeurIPS*.
- Bommasani, R., et al. (2021). On the opportunities and risks of foundation models. arXiv:2108.07258.
- Kaplan, J., et al. (2020). Scaling laws for neural language models. arXiv:2001.08361.
- Wei, J., et al. (2022). Emergent abilities of large language models. *TMLR*. (cf. Schaeffer et al., 2023, mirage critique, *NeurIPS*.)
- Gilardi, F., Alizadeh, M., & Kubli, M. (2023). ChatGPT outperforms crowd workers for text-annotation tasks. *PNAS*, 120(30).
- Törnberg, P. (2023). ChatGPT-4 outperforms experts and crowd workers in annotating political Twitter messages. arXiv:2304.06588.
- Rathje, S., et al. (2024). GPT is an effective tool for multilingual psychological text analysis. *PNAS*, 121(34).
- Ziems, C., et al. (2024). Can large language models transform computational social science? *Computational Linguistics*, 50(1).
- Jacobs, A. Z., & Wallach, H. (2021). Measurement and fairness. *FAccT*.
- Aroyo, L., & Welty, C. (2015). Truth is a lie: Crowd truth and the seven myths of human annotation. *AI Magazine*, 36(1).
- Plank, B. (2022). The "problem" of human label variation. *EMNLP*.

*Adjacent lineages*
- Grimmer, J., & Stewart, B. M. (2013). Text as data. *Political Analysis*, 21(3).
- Schwartz, H. A., et al. (2013). Personality, gender, and age in the language of social media. *PLOS ONE*, 8(9).
- Kosinski, M., Stillwell, D., & Graepel, T. (2013). Private traits and attributes are predictable from digital records of human behavior. *PNAS*, 110(15).
- Eichstaedt, J. C., et al. (2018). Facebook language predicts depression in medical records. *PNAS*, 115(44).
- Bedi, G., et al. (2015). Automated analysis of free speech predicts psychosis onset in high-risk youths. *npj Schizophrenia*, 1.
- Atkins, D. C., et al. (2014). Scaling up the evaluation of psychotherapy: evaluating motivational interviewing fidelity via statistical text classification. *Implementation Science*, 9(49).

*AI-era cognition studies (the measurement gap)*
- Lee, H.-P., et al. (2025). The impact of generative AI on critical thinking. *CHI 2025*. [verified 2026-07-15]
- Kosmyna, N., et al. (2025). Your brain on ChatGPT. MIT Media Lab. [verified 2026-07-15]
- Sparrow, B., Liu, J., & Wegner, D. M. (2011). Google effects on memory. *Science*, 333.
- Risko, E. F., & Gilbert, S. J. (2016). Cognitive offloading. *Trends in Cognitive Sciences*, 20(9).
- Messeri, L., & Crockett, M. J. (2024). Artificial intelligence and illusions of understanding in scientific research. *Nature*, 627.

*Citation status*: Rathje 2024 and Törnberg 2023 bibliographic details verified by web search 2026-07-15; Lee 2025 and Kosmyna 2025 verified earlier same day. Classic references cited from established knowledge — spot-check volume/page numbers before external submission.
