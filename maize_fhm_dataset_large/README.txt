SYNTHETIC MAIZE SEEDLING DATASET — Female / Hybrid / Male labeling scheme
============================================================================

This dataset uses the same class naming convention as the reference project:
Female (detasseled seed-parent line), Male (pollinator line), and Hybrid (F1 cross).

This is procedurally generated placeholder data, NOT real maize photography.
Visual profiles reflect real maize breeding biology: the Hybrid class shows
greater vigor (taller, darker green, more leaves) reflecting real hybrid vigor
(heterosis), while Female and Male are two visually distinct inbred parent lines.

IMPORTANT AGRONOMIC NOTE FOR YOUR REPORT:
In real maize seed production, Female and Male rows are usually different
varieties selected for the cross, and the physical distinction (detasseling)
happens at flowering stage, not the seedling stage. Any real seedling-stage
Female vs Male classification would rely on subtler inherited morphological
differences between the two parent varieties, not the detasseling itself.

STRUCTURE
---------
raw/female/   100 images
raw/hybrid/   100 images
raw/male/     100 images
metadata.csv  one row per image
class_labels.json  matches reference project format: {Female:0, Hybrid:1, Male:2}
