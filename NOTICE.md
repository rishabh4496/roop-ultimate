# Notices and attribution

## Independence

**Roop Ultimate is an independent project.** It is not affiliated with,
sponsored by, endorsed by, or connected to any other project, organisation or
individual, including the upstream projects credited below. Those projects are
credited because this software began as a derivative of their code, which their
licence requires and which is a statement of history — not of any ongoing
relationship. Do not direct support requests, bug reports or any other
correspondence about Roop Ultimate to them.

Roop Ultimate has its own repository, its own launcher, its own React user
interface, its own release numbering, and no runtime or install-time dependency
on any upstream project's repository, releases or infrastructure.

## Licence

Roop Ultimate is licensed under the **GNU Affero General Public License,
version 3** — see [`LICENSE`](LICENSE) for the full text.

This is not a choice; it is inherited. The code this project grew from is
AGPL-3.0, and the AGPL requires that derivative works be licensed as a whole
under the same terms, with the original copyright notices preserved. Renaming a
project, rewriting its interface or replacing large parts of it does not change
that, and removing the notices would be a licence violation rather than a
rebrand.

What the AGPL does **not** do, and what is worth being clear about given how
this project is distributed:

- It does not require you to publish anything. You may keep this repository
  private and modify it privately for as long as you like.
- Its obligations attach when you **convey** the software — that is, when you
  give a copy to someone else. Adding a collaborator to a private repository is
  conveying it to that person. They receive the same AGPL rights you have.
- Section 13 additionally applies if you let people **interact with it over a
  network**: those users must be offered the corresponding source. Running it
  locally for yourself does not trigger this.
- You cannot add licence terms that restrict what a recipient may do with the
  code. You are, however, under no obligation to give a copy to anyone in the
  first place — access control is what limits distribution here, not the
  licence.

## Attribution

Roop Ultimate derives, through successive forks, from:

- **roop** — <https://github.com/s0md3v/roop>, AGPL-3.0
- **roop-unleashed** — <https://github.com/C0untFloyd/roop-unleashed>, AGPL-3.0

Copyright in those portions remains with their respective authors. Extensive
modifications, additions and replacements in this repository are copyright the
Roop Ultimate contributors and are released under the same licence.

### Statement of changes (AGPL-3.0 §5(a))

This version differs substantially from the code it derives from. In summary:

- A new React user interface (`react-ui/`) and a FastAPI backend (`app/api.py`),
  replacing the original Gradio interface as the default entry point. The Gradio
  interface remains in `app/ui/` as a frozen legacy path.
- A Pinokio launcher (`install.js`, `start.js`, `pinokio.js`) that installs and
  runs the application without reference to any upstream repository.
- A rebuilt face-detection stage, including temporal detection, identity
  tracking, orientation handling and multi-face demarcation.
- Additional and reworked swap models, mask engines and enhancers, with a
  session-pooling layer for TensorRT multi-context execution.
- A measurement and benchmarking suite under `app/tests/`.

The commit history in this repository is the authoritative and complete record
of what changed.

## Third-party models and weights

This software downloads and runs machine-learning models that are **not** part of
this project and are **not** covered by this project's licence. Each carries its
own licence and its own restrictions, some of which prohibit commercial use.
Obtaining and complying with those terms is the responsibility of whoever runs
the software.

### VisoMaster

The `InStyleSwapper256` (Version A / B / C) and `CSCS` swap models, together with
CSCS's `cscs_arcface_model` recognizer and `cscs_id_adapter`, are obtained from
the VisoMaster asset release at
<https://github.com/visomaster/visomaster-assets>. VisoMaster is licensed
**GPL-3.0**; this project is AGPL-3.0, with which GPL-3.0 material may be
combined. The model files themselves are downloaded at run time and are not
redistributed here.

The integration code in this repository — the registry entries, the `cscs_dual`
identity path and the alignment/normalization values — was written for this
project against the published models. The parameters that upstream documentation
did not settle (InStyleSwapper's alignment template, and its input/output
normalization) were determined by measurement here; the evidence is recorded
beside the entries in `app/roop/processors/FaceSwapInsightFace.py`.

## Intended use

This software performs face replacement in images and video. Use it only on
material you have the right to use, and only with the informed consent of the
people whose likenesses are involved. Do not use it to create sexual content of
anyone without their consent, to impersonate real people, to produce
misinformation, or for anything unlawful in your jurisdiction. Many
jurisdictions specifically regulate synthetic media; where required, label
output as synthetic.
