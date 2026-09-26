# MOVi-A axis-overlay versus heading-up symbolic diagnostic

This matched, object-relative-only task repeats the clean controls and the
independent detection, centroid, and orientation sweeps from the original
representation-noise diagnostic.

- **RGB overlay:** the RGB scene contains only four arrows starting at the
  reference object.  Their endpoint labels are `front`, `right`, `back`, and
  `left`.  It contains no entity-name labels, symbols, boxes, or legend.
- **Symbolic:** the blank visual contains only a reference symbol and
  candidate symbols.  The prompt supplies the symbol-to-entity legend, as in
  the original symbolic task.  There is no heading arrow or axis text in the
  symbolic image: the whole layout is rotated so the reference front is
  image-up.

Heading error rotates all RGB axes.  For symbolic, it instead corrupts the
heading used for image-up alignment, rotating the whole symbol layout.
Detection and centroid noise change only the symbolic intermediate map;
the RGB condition deliberately retains its raw-pixel redundancy and intrinsic
axis overlay.

Use `KUBRIC_RELATIVE_REPRESENTATION_SUITE=all` and set
`KUBRIC_RELATIVE_REPRESENTATION_DEBUG=1` to save exact final prompt images.
