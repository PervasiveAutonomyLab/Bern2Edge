"""
bern2edge
---------
Shared library for the Bern2Edge experiments: knowledge-distillation of
Bernstein-activation student networks and their extraction into interpretable,
hardware-friendly symbolic rules.

Modules:
  * bernstein       - the Bernstein polynomial activation layer.
  * models          - teacher / student network definitions (FCModel, *TeacherMLP).
  * data            - dataset loaders and preprocessing (Adult, Cover Type, HIGGS,
                      MAGIC, ACS Income).
  * kdtrain         - knowledge-distillation training loops.
  * train_utils     - plain (non-KD) train / eval helpers.
  * rule_extraction - dataset-agnostic rule-extraction package (see its own docstring).

Nothing is re-exported here: `data` pulls in scikit-learn and the dataset
fetchers, so importing the package stays cheap. Import the module you need:

    from bern2edge.models import FCModel
    from bern2edge.rule_extraction import ExtractionConfig, extract_rules

The per-dataset drivers live outside this package, in Adult/, cover_type/,
higgs_small/, MAGIC/ and ACS/.
"""
